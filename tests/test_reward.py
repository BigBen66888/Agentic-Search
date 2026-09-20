from search_r1_refine.rl.reward import score_trajectory

def test_binary_reward():
    good = score_trajectory("<answer>Paris</answer>", ["Paris"], mode="binary")
    bad = score_trajectory("<answer>London</answer>", ["Paris"], mode="binary")
    assert good["total"] == 1 and bad["total"] == 0

def test_multi_reward():
    text = "<think>x</think><search>capital</search><information>Paris is capital</information><answer>Paris</answer>"
    score = score_trajectory(text, ["Paris"])
    assert score["answer"] > 0 and score["evidence"] > 0 and score["format"] > 0

