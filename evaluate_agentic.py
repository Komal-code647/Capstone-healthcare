import json

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

from rouge_score import rouge_scorer

from app.crew.crew import run_healthcare_crew


# Load evaluation dataset
with open("evaluation_dataset.json", "r") as f:
    dataset = json.load(f)


# Initialize ROUGE scorer
rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rougeL'],
    use_stemmer=True
)


# BLEU smoothing
smoothie = SmoothingFunction().method1


# Metric storage
bleu_scores = []
rouge1_scores = []
rougeL_scores = []
relevance_scores = []


# Evaluation loop
for item in dataset:

    try:

        question = item["question"]
        ground_truth = item["ground_truth"]

        print("\n===========================")
        print(f"Question: {question}")

        # Generate Agentic RAG response
        generated_answer = str(
            run_healthcare_crew(question)
        )

        print(f"\nGenerated Answer:\n{generated_answer}")

        # ---------------- BLEU ----------------
        bleu = sentence_bleu(
            [ground_truth.split()],
            generated_answer.split(),
            smoothing_function=smoothie
        )

        bleu_scores.append(bleu)

        # ---------------- ROUGE ----------------
        rouge_result = rouge.score(
            ground_truth,
            generated_answer
        )

        rouge1 = rouge_result['rouge1'].fmeasure
        rougeL = rouge_result['rougeL'].fmeasure

        rouge1_scores.append(rouge1)
        rougeL_scores.append(rougeL)

        # ---------------- Relevance ----------------
        overlap = len(
            set(ground_truth.lower().split())
            &
            set(generated_answer.lower().split())
        )

        relevance = overlap / max(
            len(set(ground_truth.lower().split())),
            1
        )

        relevance_scores.append(relevance)

        # ---------------- Print Metrics ----------------
        print(f"\nBLEU Score: {bleu:.4f}")
        print(f"ROUGE-1 Score: {rouge1:.4f}")
        print(f"ROUGE-L Score: {rougeL:.4f}")
        print(f"Relevance Score: {relevance:.4f}")

    except Exception as e:

        print("\nERROR during evaluation:")
        print(e)

        continue


# ---------------- Final Results ----------------

if len(bleu_scores) > 0:

    print("\n========= FINAL RESULTS =========")

    print(
        f"Average BLEU: "
        f"{sum(bleu_scores)/len(bleu_scores):.4f}"
    )

    print(
        f"Average ROUGE-1: "
        f"{sum(rouge1_scores)/len(rouge1_scores):.4f}"
    )

    print(
        f"Average ROUGE-L: "
        f"{sum(rougeL_scores)/len(rougeL_scores):.4f}"
    )

    print(
        f"Average Relevance: "
        f"{sum(relevance_scores)/len(relevance_scores):.4f}"
    )

else:

    print("\nNo evaluation results generated.")
