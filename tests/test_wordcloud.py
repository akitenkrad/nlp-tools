from nlp_tools.utils.utils import word_cloud, WordCloudMask

def test_word_cloud():
    input_text = "testa testb testc testd teste"
    word_cloud(input_text, "tests/output/test_wc.png", WordCloudMask.CIRCLE)
