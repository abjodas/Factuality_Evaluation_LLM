CONSISTENCY_COT_PROMPT = \
"""Decide if the following summary is consistent with the corresponding article. 
    Note that consistency means all information in the summary is supported by the article.
    Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
    <Article>
    {article}
    </Article>

    <Summary>
    {summary}
    </Summary>

    Answer:"""

CLAIM_PROMPT = \
"""
<Article>
{article}
</Article>

<Claim>
{summary}
</Claim>
"""