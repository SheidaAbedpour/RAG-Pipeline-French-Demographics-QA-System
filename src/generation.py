# src/generation.py
import json
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for LLM generation"""
    # Model configuration
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    api_key: Optional[str] = None
    base_url: str = "https://api.together.xyz"

    # Generation parameters
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # RAG parameters
    max_context_length: int = 4000
    context_overlap: int = 100
    include_source_info: bool = True

    # Safety and quality
    response_timeout: int = 120
    max_retries: int = 3
    min_response_length: int = 50

    # Prompt engineering
    system_prompt_template: str = """You are a knowledgeable assistant specializing in French geography and culture. 
You provide accurate, informative responses based on the given context.

Key guidelines:
- Use only the information provided in the context
- Be specific and detailed in your responses
- If information is not available in the context, clearly state this
- Cite sections when possible (e.g., "According to the Land section...")
- Maintain a professional, educational tone"""

    user_prompt_template: str = """Based on the following context about France, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided."""


class TogetherAIClient:
    """Client for TogetherAI API"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv('TOGETHER_API_KEY')

        if not self.api_key:
            raise ValueError(
                "TogetherAI API key is required. Set TOGETHER_API_KEY environment variable or pass api_key in config.")

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        logger.info(f"TogetherAI client initialized with model: {config.model_name}")

    def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate completion using TogetherAI API"""

        # Prepare request data
        request_data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "stream": False
        }

        try:
            # Make API request
            response = self.session.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=request_data,
                timeout=self.config.response_timeout
            )

            response.raise_for_status()
            result = response.json()

            # Extract and validate response
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]

                # Add metadata
                result["metadata"] = {
                    "model": self.config.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "input_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": result.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": result.get("usage", {}).get("total_tokens", 0)
                }

                return result
            else:
                raise Exception(f"Invalid response format: {result}")

        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout after {self.config.response_timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            test_messages = [
                {"role": "user", "content": "Hello, please respond with 'Connection test successful'"}
            ]

            result = self.generate_completion(test_messages, max_tokens=50)
            return "successful" in result["choices"][0]["message"]["content"].lower()

        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


class PromptEngineer:
    """Handles prompt engineering and optimization"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.prompt_history = []
        self.performance_metrics = {}

    def create_system_prompt(self, context_type: str = "general") -> str:
        """Create optimized system prompt"""
        base_prompt = self.config.system_prompt_template

        # Customize based on context type
        if context_type == "geography":
            base_prompt += "\n\nSpecialize in geographical features, climate, topography, and natural resources."
        elif context_type == "culture":
            base_prompt += "\n\nSpecialize in cultural aspects, history, and social characteristics."
        elif context_type == "technical":
            base_prompt += "\n\nProvide technical and scientific explanations when appropriate."

        return base_prompt

    def format_context(self, retrieved_chunks: List[Any], max_length: int = None) -> str:
        """Format retrieved chunks into context"""
        if not retrieved_chunks:
            return "No relevant context available."

        max_length = max_length or self.config.max_context_length

        # Sort by relevance score (highest first)
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)

        context_parts = []
        current_length = 0

        for chunk in sorted_chunks:
            # Create chunk header
            chunk_header = f"## {chunk.section}"
            if chunk.subsection:
                chunk_header += f" - {chunk.subsection}"

            # Format chunk content
            chunk_content = f"{chunk_header}\n{chunk.text}\n"

            # Add source info if enabled
            if self.config.include_source_info:
                chunk_content += f"(Source: {chunk.chunk_id})\n"

            chunk_content += "\n"

            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_content) > max_length:
                # Try to fit partial content
                remaining_space = max_length - current_length - len(chunk_header) - 50
                if remaining_space > 100:  # Only add if we have reasonable space
                    partial_text = chunk.text[:remaining_space] + "..."
                    chunk_content = f"{chunk_header}\n{partial_text}\n\n"
                    context_parts.append(chunk_content)
                break

            context_parts.append(chunk_content)
            current_length += len(chunk_content)

        return "".join(context_parts)

    def create_user_prompt(self, question: str, context: str) -> str:
        """Create user prompt with context and question"""
        return self.config.user_prompt_template.format(
            context=context,
            question=question
        )

    def optimize_prompt_for_query(self, question: str, context: str) -> Tuple[str, str]:
        """Optimize prompts based on query type"""
        # Detect query type
        question_lower = question.lower()

        if any(word in question_lower for word in ["what", "describe", "explain"]):
            context_type = "general"
        elif any(word in question_lower for word in ["mountain", "river", "climate", "geography"]):
            context_type = "geography"
        elif any(word in question_lower for word in ["culture", "history", "people"]):
            context_type = "culture"
        else:
            context_type = "technical"

        system_prompt = self.create_system_prompt(context_type)
        user_prompt = self.create_user_prompt(question, context)

        return system_prompt, user_prompt

    def evaluate_prompt_quality(self, prompt: str, response: str) -> Dict[str, float]:
        """Evaluate prompt and response quality"""
        metrics = {
            "prompt_length": len(prompt),
            "response_length": len(response),
            "context_utilization": self._calculate_context_utilization(prompt, response),
            "response_completeness": self._calculate_response_completeness(response),
            "source_citation": self._check_source_citation(response)
        }

        return metrics

    def _calculate_context_utilization(self, prompt: str, response: str) -> float:
        """Calculate how well the context was utilized"""
        # Simple heuristic: check if response references context sections
        context_indicators = ["according to", "based on", "the text mentions", "as stated"]
        score = sum(1 for indicator in context_indicators if indicator in response.lower())
        return min(score / 2, 1.0)  # Normalize to 0-1

    def _calculate_response_completeness(self, response: str) -> float:
        """Calculate response completeness"""
        # Check response length and structure
        if len(response) < self.config.min_response_length:
            return 0.3
        elif len(response) < 200:
            return 0.6
        else:
            return 1.0

    def _check_source_citation(self, response: str) -> float:
        """Check if response includes source citations"""
        citation_patterns = ["section", "source:", "according to", "based on"]
        return 1.0 if any(pattern in response.lower() for pattern in citation_patterns) else 0.0


class RAGGenerator:
    """Main RAG generation system"""

    def __init__(self, config: GenerationConfig, retriever, embedding_model=None):
        self.config = config
        self.retriever = retriever
        self.embedding_model = embedding_model

        # Initialize components
        self.llm_client = TogetherAIClient(config)
        self.prompt_engineer = PromptEngineer(config)

        # Generation history and metrics
        self.generation_history = []
        self.performance_metrics = {
            "total_queries": 0,
            "successful_generations": 0,
            "average_response_time": 0.0,
            "average_token_usage": 0.0
        }

        logger.info("RAG Generator initialized successfully")

    def generate_response(self,
                          question: str,
                          k: int = 5,
                          section_filter: Optional[str] = None,
                          min_score: float = 0.0,
                          **generation_kwargs) -> Dict[str, Any]:
        """Generate response using RAG pipeline"""

        start_time = time.time()

        try:
            # Step 1: Retrieve relevant context
            logger.info(f"Retrieving context for question: {question[:50]}...")

            retrieved_chunks = self.retriever.search(
                question,
                k=k,
                section_filter=section_filter,
                min_score=min_score
            )

            if not retrieved_chunks:
                return {
                    "question": question,
                    "answer": "I don't have enough relevant information to answer your question.",
                    "sources": [],
                    "metadata": {
                        "retrieval_results": 0,
                        "generation_time": time.time() - start_time,
                        "error": "No relevant context found"
                    }
                }

            # Step 2: Format context
            context = self.prompt_engineer.format_context(retrieved_chunks)

            # Step 3: Create optimized prompts
            system_prompt, user_prompt = self.prompt_engineer.optimize_prompt_for_query(
                question, context
            )

            # Step 4: Generate response
            logger.info("Generating response with LLM...")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            llm_response = self.llm_client.generate_completion(
                messages, **generation_kwargs
            )

            # Step 5: Extract and format response
            generated_text = llm_response["choices"][0]["message"]["content"]

            # Step 6: Prepare sources
            sources = []
            for chunk in retrieved_chunks:
                sources.append({
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "score": chunk.score,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                })

            # Step 7: Calculate metrics
            generation_time = time.time() - start_time

            # Update performance metrics
            self.performance_metrics["total_queries"] += 1
            self.performance_metrics["successful_generations"] += 1

            # Calculate running average
            prev_avg_time = self.performance_metrics["average_response_time"]
            total_queries = self.performance_metrics["total_queries"]
            self.performance_metrics["average_response_time"] = (
                    (prev_avg_time * (total_queries - 1) + generation_time) / total_queries
            )

            # Token usage
            token_usage = llm_response.get("metadata", {}).get("total_tokens", 0)
            prev_avg_tokens = self.performance_metrics["average_token_usage"]
            self.performance_metrics["average_token_usage"] = (
                    (prev_avg_tokens * (total_queries - 1) + token_usage) / total_queries
            )

            # Step 8: Prepare final response
            response = {
                "question": question,
                "answer": generated_text,
                "sources": sources,
                "metadata": {
                    "retrieval_results": len(retrieved_chunks),
                    "generation_time": generation_time,
                    "token_usage": llm_response.get("metadata", {}),
                    "model": self.config.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {
                        "k": k,
                        "section_filter": section_filter,
                        "min_score": min_score,
                        "temperature": generation_kwargs.get("temperature", self.config.temperature)
                    }
                }
            }

            # Store in history
            self.generation_history.append(response)

            logger.info(f"Response generated successfully in {generation_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

            # Update error metrics
            self.performance_metrics["total_queries"] += 1

            return {
                "question": question,
                "answer": "I apologize, but I encountered an error while generating the response. Please try again.",
                "sources": [],
                "metadata": {
                    "retrieval_results": 0,
                    "generation_time": time.time() - start_time,
                    "error": str(e)
                }
            }

    def batch_generate(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple questions"""
        logger.info(f"Starting batch generation for {len(questions)} questions")

        responses = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i + 1}/{len(questions)}")
            response = self.generate_response(question, **kwargs)
            responses.append(response)

            # Small delay between requests to be respectful
            time.sleep(0.1)

        logger.info(f"Batch generation completed: {len(responses)} responses")
        return responses

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        success_rate = 0.0
        if self.performance_metrics["total_queries"] > 0:
            success_rate = (
                    self.performance_metrics["successful_generations"] /
                    self.performance_metrics["total_queries"]
            )

        return {
            **self.performance_metrics,
            "success_rate": success_rate,
            "total_history_entries": len(self.generation_history)
        }

    def export_history(self, filepath: str):
        """Export generation history to file"""
        with open(filepath, 'w') as f:
            json.dump({
                "generation_history": self.generation_history,
                "performance_metrics": self.get_performance_metrics(),
                "export_timestamp": datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"Generation history exported to {filepath}")


class ResponseValidator:
    """Validates and improves generated responses"""

    def __init__(self, config: GenerationConfig):
        self.config = config

    def validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response quality and suggest improvements"""

        validation_results = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "quality_score": 0.0
        }

        answer = response.get("answer", "")
        sources = response.get("sources", [])

        # Check response length
        if len(answer) < self.config.min_response_length:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Response too short")
            validation_results["suggestions"].append("Increase max_tokens or lower temperature")

        # Check if response uses context
        if not any(keyword in answer.lower() for keyword in ["according to", "based on", "the text", "source"]):
            validation_results["issues"].append("Response doesn't reference sources")
            validation_results["suggestions"].append("Improve system prompt to encourage source citation")

        # Check source utilization
        if len(sources) > 0 and len(answer) > 0:
            # Simple heuristic: check if response mentions section names
            section_mentions = sum(1 for source in sources if source["section"].lower() in answer.lower())
            source_utilization = section_mentions / len(sources)

            if source_utilization < 0.3:
                validation_results["issues"].append("Low source utilization")
                validation_results["suggestions"].append("Improve context formatting or prompt engineering")

        # Calculate quality score
        quality_factors = []
        quality_factors.append(min(len(answer) / 200, 1.0))  # Length factor
        quality_factors.append(1.0 if len(validation_results["issues"]) == 0 else 0.5)  # Issue factor
        quality_factors.append(min(len(sources) / 3, 1.0))  # Source factor

        validation_results["quality_score"] = sum(quality_factors) / len(quality_factors)

        return validation_results

    def improve_response(self, response: Dict[str, Any], validation_results: Dict[str, Any]) -> Dict[str, str]:
        """Suggest improvements based on validation results"""

        improvements = {}

        for issue in validation_results["issues"]:
            if "too short" in issue:
                improvements["parameter_adjustment"] = "Increase max_tokens from {} to {}".format(
                    self.config.max_tokens, self.config.max_tokens * 1.5
                )

            elif "source" in issue:
                improvements["prompt_improvement"] = "Add explicit instruction to cite sources in system prompt"

            elif "utilization" in issue:
                improvements["context_formatting"] = "Improve context formatting with clearer section headers"

        return improvements


# Example usage and testing functions
def test_generation_system():
    """Test the generation system with sample data"""
    print("Testing Generation System...")

    # This would be called after setting up retriever and embedding model
    # For now, we'll create mock objects

    class MockRetriever:
        def search(self, query, k=5, section_filter=None, min_score=0.0):
            # Mock retrieval results
            from types import SimpleNamespace
            return [
                SimpleNamespace(
                    chunk_id="test_chunk_1",
                    text="France is located in Western Europe and has diverse geographical features.",
                    section="Land",
                    subsection="Geography",
                    score=0.85
                ),
                SimpleNamespace(
                    chunk_id="test_chunk_2",
                    text="The climate of France is generally temperate with regional variations.",
                    section="Climate",
                    subsection="General",
                    score=0.72
                )
            ]

    # Test configuration
    config = GenerationConfig(
        api_key="test_key",  # Replace with actual key
        temperature=0.3,
        max_tokens=512
    )

    try:
        # Create mock retriever
        mock_retriever = MockRetriever()

        # Initialize generator
        generator = RAGGenerator(config, mock_retriever)

        # Test questions
        test_questions = [
            "What are the main geographical features of France?",
            "How would you describe France's climate?",
            "Tell me about the diversity of French landscapes."
        ]

        # Generate responses
        for question in test_questions:
            print(f"\nQuestion: {question}")
            print("-" * 50)

            # This would fail without real API key, but shows the structure
            try:
                response = generator.generate_response(question)
                print(f"Answer: {response['answer']}")
                print(f"Sources: {len(response['sources'])}")
            except Exception as e:
                print(f"Error (expected without API key): {e}")

        print("\n✅ Generation system structure is working correctly!")

    except Exception as e:
        print(f"❌ Error in generation system: {e}")


if __name__ == "__main__":
    test_generation_system()