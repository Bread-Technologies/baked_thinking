"""
Stimulus configuration for Metacognitive Baking.
Defines the strategies for generating questions that test confidence calibration.
"""

STIMULUS_CONFIG = {
    # 1. Factual Knowledge (Should be <confident>)
    "factual_knowledge": {
        "type": "oneshot_qs",
        "model": "claude-3-5-sonnet-20240620",
        "numq": 200,
        "template": "Generate diverse factual questions with clear, unambiguous answers. Topics: Science, History, Geography, Math. Avoid controversial or changing facts.",
        "examples": [
            "What is the atomic number of Carbon?",
            "Who was the first President of the United States?",
            "What is the square root of 144?"
        ]
    },

    # 2. False Premises (Should be <uncertain>)
    "false_premises": {
        "type": "oneshot_qs",
        "model": "claude-3-5-sonnet-20240620",
        "numq": 150,
        "template": "Generate questions that contain a subtle but factually incorrect premise. The user assumes something true that is false.",
        "examples": [
            "When did Einstein discover gravity?",
            "Who was the sole survivor of the 1987 Mars Rover disaster?",
            "What is the capital of the continent of Antarctica?"
        ]
    },

    # 3. Ambiguous Entities (Should be <uncertain>)
    "ambiguous_entities": {
        "type": "oneshot_qs",
        "model": "claude-3-5-sonnet-20240620",
        "numq": 150,
        "template": "Generate questions about entities or terms that are ambiguous without more context (e.g., common names, multiple versions).",
        "examples": [
            "Did John Smith win the election?",
            "How do I fix the engine light on my Ford?",
            "Is the sequel better than the original?"
        ]
    },

    # 4. Subjective/Opinion (Should be <uncertain>)
    "subjective_opinion": {
        "type": "oneshot_qs",
        "model": "claude-3-5-sonnet-20240620",
        "numq": 100,
        "template": "Generate questions that ask for definitive answers to subjective or matter-of-taste questions.",
        "examples": [
            "What is the best flavor of ice cream?",
            "Is Javascript better than Python?",
            "Who is the greatest musician of all time?"
        ]
    },

    # 5. Reasoning Traps (Should be <confident> after <think> correction)
    "reasoning_traps": {
        "type": "oneshot_qs",
        "model": "claude-3-5-sonnet-20240620",
        "numq": 150,
        "template": "Generate 'trick' reasoning questions that have an intuitive but wrong answer (System 1 bait), requiring System 2 thinking.",
        "examples": [
            "A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
            "Mary's father has 5 daughters: Nana, Nene, Nini, Nono. Who is the 5th daughter?"
        ]
    }
}


