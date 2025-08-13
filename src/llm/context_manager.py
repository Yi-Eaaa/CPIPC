from config.config import GLOABLE_CONFIG
from llm.agent_context import AgentContext


class ContextManager:
    def __init__(
        self, default_token_limit=GLOABLE_CONFIG["default_context_token_limit"]
    ):
        self.name2sessionidx = {}
        self.session_contexts: list[AgentContext] = []
        self.default_token_limit = default_token_limit

    def create_session(self, session_name, token_limit=None):
        assert (
            session_name not in self.name2sessionidx
        ), f"Session {session_name} already exists."
        session_idx = len(self.session_contexts)
        self.name2sessionidx[session_name] = session_idx
        if token_limit is None:
            token_limit = self.default_token_limit
        self.session_contexts.append(AgentContext(token_limit=token_limit))

    def add_context(self, session_name, added_context):
        self.session_contexts[self.get_sessionidx(session_name)].add_context(
            added_context
        )

    def get_context(self, session_name):
        return self.session_contexts[self.get_sessionidx(session_name)].get_context()

    def get_context_token_nums(self, session_name):
        return self.session_contexts[
            self.get_sessionidx(session_name)
        ].get_context_token_nums()

    def get_sessionidx(self, session_name):
        session_idx = self.name2sessionidx.get(session_name, None)
        assert (
            session_idx is not None
        ), f"Session {session_name} not found. Please create the session first."
        return session_idx


if __name__ == "__main__":
    pass
