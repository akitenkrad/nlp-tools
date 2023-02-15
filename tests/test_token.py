from utils.data import Token


def test_to_instantiate_token():
    token = Token("surface", "base", "pos_tag")

    assert token.surface == "surface"
    assert token.baes == "base"
    assert token.pos_tag == "pos_tag"


def test_to_hash_token():
    token = Token("surface", "base", "pos_tag")

    test_dict = {token: 1}
    assert test_dict is not None

    assert token == Token("surface", "base", "pos_tag")
    assert token != Token("surface", "base", "pos-tag")
