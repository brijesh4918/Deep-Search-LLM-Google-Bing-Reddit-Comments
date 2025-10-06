from typing import Dict, Any, List


class PromptFactory:
    """Centralized repository for all LLM prompt templates used by the Deep Research Agent."""

    # -------------------------------------------------------------------------
    # 🧩 Reddit URL Analysis
    # -------------------------------------------------------------------------
    @staticmethod
    def reddit_url_selector_system() -> str:
        """System prompt: Select the most relevant Reddit URLs."""
        return (
            "You are a skilled social media content analyst. Your task is to review Reddit search results "
            "and identify URLs of posts that offer strong informational value for answering the user's query.\n\n"
            "Focus on posts that:\n"
            "- Closely address the user's question\n"
            "- Include detailed discussions, expert advice, or data-backed claims\n"
            "- Show strong community engagement (upvotes/comments)\n"
            "- Contribute diverse or unique perspectives\n\n"
            "Return a structured list containing the most relevant Reddit URLs."
        )

    @staticmethod
    def reddit_url_selector_user(question: str, reddit_output: str) -> str:
        """User prompt: Provide Reddit search results for filtering."""
        return (
            f"User Question: {question}\n\n"
            f"Reddit Search Results:\n{reddit_output}\n\n"
            "Analyze the above Reddit results and identify posts most useful for addressing the question."
        )

    # -------------------------------------------------------------------------
    # 🌐 Google Search Analysis
    # -------------------------------------------------------------------------
    @staticmethod
    def google_analysis_system() -> str:
        """System prompt: Analyze Google search results."""
        return (
            "You are an expert research analyst. Review the provided Google search results and extract key insights "
            "that help answer the user's question.\n\n"
            "Focus on:\n"
            "- Verified facts and authoritative sources (official docs, academic papers, reputable sites)\n"
            "- Important statistics, dates, and figures\n"
            "- Conflicting viewpoints or data inconsistencies\n\n"
            "Deliver a concise, factual summary highlighting the most relevant insights."
        )

    @staticmethod
    def google_analysis_user(question: str, google_output: str) -> str:
        """User prompt: Provide Google search results."""
        return (
            f"Question: {question}\n\n"
            f"Google Search Results:\n{google_output}\n\n"
            "Analyze these Google results and extract key findings that help answer the question."
        )

    # -------------------------------------------------------------------------
    # 🧠 Bing Search Analysis
    # -------------------------------------------------------------------------
    @staticmethod
    def bing_analysis_system() -> str:
        """System prompt: Analyze Bing search results."""
        return (
            "You are an analytical researcher. Review Bing search results to uncover complementary insights that "
            "enrich the understanding of the user's query.\n\n"
            "Focus on:\n"
            "- Technical articles and enterprise perspectives\n"
            "- Alternative viewpoints not present in other sources\n"
            "- Recent news updates and announcements\n"
            "- Microsoft ecosystem or industry-specific insights\n\n"
            "Summarize the distinct and useful information found in these results."
        )

    @staticmethod
    def bing_analysis_user(question: str, bing_output: str) -> str:
        """User prompt: Provide Bing search results."""
        return (
            f"Question: {question}\n\n"
            f"Bing Search Results:\n{bing_output}\n\n"
            "Analyze these Bing results and highlight insights that complement findings from other sources."
        )

    # -------------------------------------------------------------------------
    # 💬 Reddit Discussion Analysis
    # -------------------------------------------------------------------------
    @staticmethod
    def reddit_discussion_system() -> str:
        """System prompt: Analyze Reddit discussion threads."""
        return (
            "You are a specialist in understanding online community discussions. Review Reddit posts and comments "
            "to extract practical user experiences and collective opinions.\n\n"
            "Focus on:\n"
            "- Real user experiences and feedback\n"
            "- Popular consensus or recurring sentiments\n"
            "- Useful advice, debates, and diverse perspectives\n"
            "- Direct quotes from posts (use quotation marks and mention subreddit if available)\n\n"
            "Provide a balanced summary capturing both positive and negative experiences."
        )

    @staticmethod
    def reddit_discussion_user(
        question: str, reddit_output: str, post_data: list
    ) -> str:
        """User prompt: Provide Reddit data for analysis."""
        return (
            f"Question: {question}\n\n"
            f"Reddit Search Results:\n{reddit_output}\n\n"
            f"Detailed Reddit Post Data:\n{post_data}\n\n"
            "Analyze the Reddit content and extract community insights, common opinions, and real-world experiences."
        )

    # -------------------------------------------------------------------------
    # 🧩 Final Answer Synthesis
    # -------------------------------------------------------------------------
    @staticmethod
    def synthesis_system() -> str:
        """System prompt: Combine insights from all sources."""
        return (
            "You are a professional research synthesizer. Combine the findings from Google, Bing, and Reddit analyses "
            "to create a unified, well-reasoned summary.\n\n"
            "Your response should:\n"
            "- Integrate information from all three sources\n"
            "- Identify overlapping and conflicting insights\n"
            "- Present a structured and balanced summary\n"
            "- Attribute key claims to their source type (Google, Bing, Reddit)\n"
            "- Highlight uncertainties or differing perspectives\n\n"
            "Output a clear, comprehensive synthesis that answers the question holistically."
        )

    @staticmethod
    def synthesis_user(
        question: str, google_summary: str, bing_summary: str, reddit_summary: str
    ) -> str:
        """User prompt: Provide all analyses for synthesis."""
        return (
            f"Question: {question}\n\n"
            f"Google Analysis:\n{google_summary}\n\n"
            f"Bing Analysis:\n{bing_summary}\n\n"
            f"Reddit Discussion Analysis:\n{reddit_summary}\n\n"
            "Combine these analyses into a unified, detailed response that reflects multiple perspectives."
        )


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def make_message_pair(system_msg: str, user_msg: str) -> List[Dict[str, Any]]:
    """Standardizes the message structure for LLM input."""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# -------------------------------------------------------------------------
# Message Generators (Shortcuts)
# -------------------------------------------------------------------------
def reddit_url_analysis_msgs(question: str, reddit_output: str) -> List[Dict[str, Any]]:
    """Generate messages for Reddit URL relevance analysis."""
    return make_message_pair(
        PromptFactory.reddit_url_selector_system(),
        PromptFactory.reddit_url_selector_user(question, reddit_output),
    )


def google_analysis_msgs(question: str, google_output: str) -> List[Dict[str, Any]]:
    """Generate messages for Google search result analysis."""
    return make_message_pair(
        PromptFactory.google_analysis_system(),
        PromptFactory.google_analysis_user(question, google_output),
    )


def bing_analysis_msgs(question: str, bing_output: str) -> List[Dict[str, Any]]:
    """Generate messages for Bing search result analysis."""
    return make_message_pair(
        PromptFactory.bing_analysis_system(),
        PromptFactory.bing_analysis_user(question, bing_output),
    )


def reddit_discussion_msgs(
    question: str, reddit_output: str, post_data: list
) -> List[Dict[str, Any]]:
    """Generate messages for Reddit discussion thread analysis."""
    return make_message_pair(
        PromptFactory.reddit_discussion_system(),
        PromptFactory.reddit_discussion_user(question, reddit_output, post_data),
    )


def synthesis_msgs(
    question: str, google_summary: str, bing_summary: str, reddit_summary: str
) -> List[Dict[str, Any]]:
    """Generate messages for final synthesis and answer generation."""
    return make_message_pair(
        PromptFactory.synthesis_system(),
        PromptFactory.synthesis_user(question, google_summary, bing_summary, reddit_summary),
    )