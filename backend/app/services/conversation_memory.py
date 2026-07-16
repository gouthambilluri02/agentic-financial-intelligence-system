class ConversationMemory:
    """
    Stores conversation context between user questions.
    """

    def __init__(self):
        self.last_question = None
        self.last_companies = []
        self.last_intent = None

    def update(
        self,
        question: str,
        companies: list[str],
        intent: str,
    ):
        self.last_question = question
        self.last_companies = companies
        self.last_intent = intent

    def get_last_question(self):
        return self.last_question

    def get_last_companies(self):
        return self.last_companies

    def get_last_intent(self):
        return self.last_intent

    def clear(self):
        self.last_question = None
        self.last_companies = []
        self.last_intent = None

if __name__ == "__main__":

    memory = ConversationMemory()

    memory.update(
        question="Compare Apple and Microsoft revenue.",
        companies=["Apple", "Microsoft"],
        intent="comparison",
    )

    print("Last Question:")
    print(memory.get_last_question())

    print("\nCompanies:")
    print(memory.get_last_companies())

    print("\nIntent:")
    print(memory.get_last_intent())

    memory.clear()

    print("\nAfter Clear:")
    print(memory.get_last_companies())        