from pathlib import Path
from langchain_core.language_models.chat_models import BaseChatModel

class UserProfileMemory:
    """
    Maintains a persistent distilled profile for each user.

    The profile is stored separately from conversation history as a markdown file.
    It is not a transcript. It contains stable facts, preferences, and recurring topics.
    """
    def __init__(self, user_id: str, profiles_dir: str = "memory/profiles"):
        self.user_id = self._safe_user_id(user_id)
        self.profiles_dir = Path(profiles_dir)
        self.profile_path = self.profiles_dir / f"{self.user_id}.md"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            self.profile_path.write_text(f"# User Profile\n\nUser ID: {self.user_id}\n\nNo stable user facts known yet.\n",
                                         encoding="utf-8")

    def load(self) -> str:
        """
        Load the user's persistent profile from disk.
        """
        return self.profile_path.read_text(encoding="utf-8")

    def save(self, profile_text: str) -> None:
        """
        Save the updated user profile to disk.
        """
        self.profile_path.write_text(profile_text.strip() + "\n", encoding="utf-8")

    def update_after_turn(self, llm: BaseChatModel, user_message: str, assistant_answer: str) -> None:
        """
        Ask the LLM to update the persistent user profile after each turn.

        The LLM should keep only stable, useful facts and preferences. It should
        not copy the conversation transcript.
        """
        current_profile = self.load()

        prompt = f"""
You maintain a persistent user profile.

Current profile:
{current_profile}

Latest user message:
{user_message}

Latest assistant answer:
{assistant_answer}

Update the profile if the latest turn reveals stable useful information about the user.

Rules:
- Keep the profile short and distilled.
- Do not copy the conversation transcript.
- Keep facts that may help future conversations.
- Include preferences, name, tools, environment, recurring topics, and project context.
- Do not include temporary debugging details unless they reveal a stable preference or setup.
- If nothing useful changed, return the current profile unchanged.
- Return only the full updated markdown profile.
"""
        updated_profile = llm.invoke(prompt).content
        if updated_profile and updated_profile.strip():
            self.save(updated_profile)

    def _safe_user_id(self, user_id: str) -> str:
        """
        Convert the user ID into a safe filename.
        """
        safe = "".join(char for char in user_id.strip().lower() if char.isalnum() or char in {"_", "-"})
        return safe or "default_user"