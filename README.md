# Zyro Dynamics HR Help Desk

A Retrieval-Augmented Generation (RAG) based HR Help Desk chatbot for answering questions from Zyro Dynamics' internal HR policy documents.

## Overview

The system allows employees to ask questions about HR policies such as:

- Leave and attendance
- Compensation and benefits
- Work from home
- Performance reviews
- Code of conduct
- IT and data securityD
- POSH
- Onboarding and separation
- Travel and expenses

The chatbot retrieves relevant information from the provided HR policy documents and uses a language model to generate a grounded response.

Questions outside the supported HR policy scope are refused rather than answered using general knowledge.

## Architecture

```text
HR Policy PDFs
      |
      v
PDF Text Extraction
      |
      v
Document Chunking
      |
      v
Sentence Transformer Embeddings
      |
      v
FAISS Vector Index
      |
      |
User Question
      |
      v
Scope Classification
      |
      +---- OUT OF SCOPE ----> Refusal
      |
      v
Semantic Retrieval
      |
      v
Relevant Policy Chunks
      |
      v
Grounded LLM
      |
      v
Answer + Sources
