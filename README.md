# Factuality Evaluation Using LLMs

The main goal of this project is to understand the ability of Large Language Models to judge whether a summary is consistent with a document or not. For a summary to be consistent, there must not be any extra information present in it which was not present in the document.

---
# Instructions

To run the project, simply clone or download this repository and then install the requirements using - 

`pip install -r requirements.txt`

You can then choose to run either the consistency evaluation task or the ranking task by passing "consistency" in the **--task** argument.
For consistency task, you have the option to select cogensumm, factcc, polytope, xsumfaith and summeval datasets. 
