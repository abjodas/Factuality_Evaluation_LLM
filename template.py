CONSISTENCY_COT_PROMPT = \
"""
Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article.
Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
<Article>
{article}
</Article>
<Summary>
{summary}
</Summary>
Answer:
"""

CONSISTENCY_PROMPT = \
"""
Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article.
Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
<Article>
{article}
</Article>
<Summary>
{summary}
</Summary>
Answer:
"""

RANKING_PROMPT = \
"""
Your task is to rate the factual consistency of the provided summary against the source article.A score of 1.0 means the summary is perfectly factual, containing no information that contradicts or is not supported by the article.
A score of 0.0 means the summary is completely non-factual.
Carefully read the article and the summary. Then, provide ONLY the numerical factuality score as your response. Do not add any explanation, commentary, or conversational text.
[ARTICLE]:
{article}
[SUMMARY]:
{summary}
[FACTUALITY SCORE]:
"""
SCORING_PROMPT = \
"""
Your task is to identify which summary is more
consistent and faithful to the original article
(preserving key facts without adding or omitting
important information). Respond with only: "A""
or "B"
<Article> {article} </Article>
<Summary A> {summary_a} </Summary A>
<Summary B> {summary_b} </Summary B>
Answer:
"""
SUMMEVAL_PROMPT_COT = \
"""
Do not provide any reasoning. Article: {article}\nSummary: {summary}
"""
SUMMEVAL_PROMPT = \
"""
Article: {article}\nSummary: {summary}
"""
SCORING_PROMPT_TAG = \
"""
Decide which one of the following summary is consistent with the corresponding article.
Note that consistency means all information in the summary is supported by the article.
Just give your answer in <Answer>(A or B)</Answer> tags:

<Article>
{document}
</Article>

<Summary A>
{sum_a}
</Summary A>
<Summary B>
{sum_b}
</Summary B>
"""
SCORING_PROMPT_TAG_COT = \
"""
Decide which one of the following summary is consistent with the corresponding article.
Note that consistency means all information in the summary is supported by the article.
Explain your reasoning step by step first, and then put the final answer in <Answer>(A or B)</Answer> tags:

<Article>
{document}
</Article>

<Summary A>
{sum_a}
</Summary A>
<Summary B>
{sum_b}
</Summary B>
"""