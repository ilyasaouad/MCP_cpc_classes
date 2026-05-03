def classify_patent(text: str):
    """
    Minimal CPC classifier (we will upgrade later)
    """

    # STEP 1: simple rule/LLM placeholder
    if "neural network" in text.lower():
        return ["G06N", "G06F"]

    if "image recognition" in text.lower():
        return ["G06V", "G06K"]

    return ["G06F"]