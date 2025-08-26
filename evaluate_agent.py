import os
import asyncio
import time
import pandas as pd
import requests
from dotenv import load_dotenv

# LlamaIndex evaluators require a running event loop in some environments.
# This ensures it works correctly, especially on Windows.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
# --- THIS IS THE KEY ADDITION ---
from llama_index.core import Response
from llama_index.core.schema import TextNode
# --- END ADDITION ---
from llama_index.llms.groq import Groq

# --- Configuration ---

# Load environment variables from a .env file in the same directory
load_dotenv()

# The API endpoint of your running FastAPI agent.
# Assumes the agent is running locally on port 8000.
AGENT_API_URL = "http://127.0.0.1:8000/query"

# A unique session ID for testing. This allows testing memory-related features
# if you run the script multiple times with the same ID.
USER_SESSION_ID = "evaluation-session-001"

# Initialize the LLM that will be used for the evaluation itself.
EVALUATOR_LLM = Groq(api_key=os.getenv("GROQ_API_KEY"), model="deepseek-r1-distill-llama-70b")

# --- The "Golden Dataset" ---
# EVALUATION_DATASET = [
#     {
#         "question": "What was Adobe's total revenue for Q2 FY2025?",
#         "expected_answer": "$5.87 billion"
#     },
#     {
#         "question": "Who is the President of Digital Experience?",
#         "expected_answer": "Anil Chakravarthy"
#     },
#     {
#         "question": "What was the year-over-year growth for Digital Media ARR?",
#         "expected_answer": "12.1 percent"
#     },
#     {
#         "question": "How many shares were repurchased in the quarter?",
#         "expected_answer": "8.6 million"
#     },
#     {
#         "question": "What is Project Fizzion and who was it co-developed with?",
#         "expected_answer": "An AI-powered design intelligence system co-developed with The Coca-Cola Company."
#     },
#     {
#         "question": "What was the Digital Experience segment revenue in Q2?",
#         "expected_answer": "$1.46 billion"
#     },
#     {
#         "question": "Summarize the financial targets for the upcoming Q3 FY2025.",
#         "expected_answer": "Total revenue of $5.875 to $5.925 billion, Digital Media revenue of $4.37 to $4.40 billion, and Non-GAAP EPS of $5.15 to $5.20."
#     },
#     {
#         "question": "What is the company's AI-influenced ARR contribution?",
#         "expected_answer": "Billions of dollars, and the AI book of business from AI-first products is tracking ahead of the $250 million ending ARR target by the end of fiscal 2025."
#     }
# ]

EVALUATION_DATASET = [
    # Direct fact retrieval
    {
        "question": "What was Adobe's total revenue in Q2 FY2025?",
        "expected_answer": "$5.87 billion"
    },
    {
        "question": "Who is Adobe's President of Digital Experience?",
        "expected_answer": "Anil Chakravarthy"
    },
    {
        "question": "What was the Digital Media revenue in Q2 FY2025?",
        "expected_answer": "$4.35 billion"
    },
    {
        "question": "How much Digital Media ARR did Adobe report at the end of Q2 FY2025?",
        "expected_answer": "$18.09 billion"
    },

    # Variants/paraphrased
    {
        "question": "How much did Adobe’s Experience Cloud business generate in Q2?",
        "expected_answer": "$1.46 billion"
    },
    {
        "question": "By what percentage did Digital Media ARR grow year over year?",
        "expected_answer": "12.1 percent"
    },
    {
        "question": "How many shares did Adobe repurchase in Q2 FY2025?",
        "expected_answer": "Adobe entered into a $3.50 billion share repurchase agreement, equivalent to approximately 8.6 million shares."
    },

    # Trend & reasoning
    {
        "question": "What was the year-over-year growth rate for Adobe’s total revenue in Q2 FY2025?",
        "expected_answer": "11 percent"
    },
    {
        "question": "Compare Digital Media and Digital Experience revenue growth rates in Q2 FY2025.",
        "expected_answer": "Digital Media grew 12 percent year-over-year, while Digital Experience grew 10 percent."
    },
    {
        "question": "Which customer groups drove higher subscription revenue in Q2, and by how much?",
        "expected_answer": "Business Professionals and Consumers grew 15 percent year-over-year to $1.60 billion, while Creative and Marketing Professionals grew 10 percent year-over-year to $4.02 billion."
    },

    # Summarization / strategic
    {
        "question": "Summarize Adobe’s Q3 FY2025 financial targets.",
        "expected_answer": "Revenue of $5.875–$5.925 billion, Digital Media revenue of $4.37–$4.40 billion, Digital Experience revenue of $1.45–$1.47 billion, GAAP EPS of $4.00–$4.05, and Non-GAAP EPS of $5.15–$5.20."
    },
    {
        "question": "What is Adobe’s revised full-year FY2025 revenue target?",
        "expected_answer": "$23.50 to $23.60 billion"
    },

    # AI/innovation focus
    {
        "question": "What is Project Fizzion, and who was it developed with?",
        "expected_answer": "An AI-powered design intelligence system co-developed with The Coca-Cola Company."
    },
    {
        "question": "What is the AI-first ARR target Adobe expects to surpass in FY2025?",
        "expected_answer": "$250 million"
    },
    {
        "question": "Name three AI-first products contributing to Adobe’s ARR.",
        "expected_answer": "Acrobat AI Assistant, Firefly App and Services, GenStudio for Performance Marketing"
    },

    # Contextual / entity disambiguation
    {
        "question": "Who were the key Adobe executives presenting the Q2 FY2025 earnings call?",
        "expected_answer": "Shantanu Narayen (Chair and CEO), David Wadhwani (President of Digital Media), Anil Chakravarthy (President of Digital Experience), and Dan Durn (EVP and CFO)."
    },
    {
        "question": "Which major sports leagues adopted Adobe Express in Q2?",
        "expected_answer": "MLB, the NFL, and the Premier League."
    }
]

async def evaluate_agent():
    """
    Runs the evaluation by sending questions to the agent and using LlamaIndex
    evaluators to score the responses for faithfulness and relevancy.
    """
    print("Initializing LlamaIndex evaluators...")
    faithfulness_evaluator = FaithfulnessEvaluator(llm=EVALUATOR_LLM)
    relevancy_evaluator = RelevancyEvaluator(llm=EVALUATOR_LLM)

    results_list = []
    total_questions = len(EVALUATION_DATASET)

    print(f"Starting evaluation with {total_questions} questions...")

    for i, item in enumerate(EVALUATION_DATASET):
        question = item["question"]
        print(f"\n--- Running Test {i+1}/{total_questions} ---")
        print(f"Question: {question}")

        try:
            start_time = time.time()
            
            response = requests.post(
                AGENT_API_URL,
                json={"query": question, "session_id": USER_SESSION_ID}
            )
            response.raise_for_status()
            
            latency = time.time() - start_time
            
            api_result = response.json()
            agent_response_text = api_result.get("response", "N/A")
            
            # --- THIS IS THE CORRECTED SECTION ---
            # Reconstruct the LlamaIndex objects from the JSON response
            sources_data = api_result.get("sources", [])
            source_nodes = [TextNode(text=source.get("node", {}).get("text", "")) for source in sources_data]
            source_texts = [node.text for node in source_nodes]

            # Create the LlamaIndex Response object that the evaluator expects
            agent_response_obj = Response(response=agent_response_text, source_nodes=source_nodes)
            # --- END OF CORRECTION ---
            
            print(f"Agent Response (snippet): {agent_response_text[:120]}...")
            print(f"Latency: {latency:.2f} seconds")

            # --- Perform Automated Evaluations using LlamaIndex ---
            
            # 1. Faithfulness (Hallucination Check)
            faithfulness_result = await faithfulness_evaluator.aevaluate_response(
                response=agent_response_obj, query=question
            )
            
            # 2. Relevancy (Retriever Check) - **THIS IS THE FIX**
            relevancy_result = await relevancy_evaluator.aevaluate_response(
                response=agent_response_obj, query=question
            )

            results_list.append({
                "question": question,
                "expected_answer": item["expected_answer"],
                "agent_response": agent_response_text,
                "latency_sec": f"{latency:.2f}",
                "is_faithful": faithfulness_result.passing,
                "is_relevant": relevancy_result.passing,
                "faithfulness_feedback": faithfulness_result.feedback,
                "relevancy_feedback": relevancy_result.feedback
            })

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Could not connect to agent API. {e}")
            results_list.append({ "question": question, "agent_response": f"API_ERROR: {e}", "latency_sec": "N/A"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            results_list.append({ "question": question, "agent_response": f"UNEXPECTED_ERROR: {e}", "latency_sec": "N/A"})

    return results_list

def main():
    """
    Main function to run the evaluation and display the results.
    """
    print("--- AI Agent Performance Evaluation Script ---")
    print(f"Targeting Agent API at: {AGENT_API_URL}")
    input("Please ensure your FastAPI agent is running in a separate terminal. Press Enter to start...")

    evaluation_results = asyncio.run(evaluate_agent())

    df = pd.DataFrame(evaluation_results)
    
    print("\n\n" + "="*50)
    print(" " * 15 + "EVALUATION RESULTS")
    print("="*50)
    print(df.to_string())

    if 'is_faithful' in df.columns and 'is_relevant' in df.columns:
        valid_faithfulness = df[df['is_faithful'].isin([True, False])]['is_faithful']
        valid_relevancy = df[df['is_relevant'].isin([True, False])]['is_relevant']
        
        avg_faithfulness = valid_faithfulness.mean() * 100 if not valid_faithfulness.empty else 0
        avg_relevancy = valid_relevancy.mean() * 100 if not valid_relevancy.empty else 0
        
        valid_latencies = pd.to_numeric(df['latency_sec'], errors='coerce').dropna()
        avg_latency = valid_latencies.mean() if not valid_latencies.empty else 0
        
        print("\n\n" + "="*50)
        print(" " * 15 + "SUMMARY STATISTICS")
        print("="*50)
        print(f"Average Faithfulness Score: {avg_faithfulness:.2f}%")
        print(f"Average Source Relevancy Score: {avg_relevancy:.2f}%")
        print(f"Average Latency: {avg_latency:.2f} seconds")
        print("="*50)
    else:
        print("\n\nCould not compute summary statistics due to errors.")

if __name__ == "__main__":
    main()