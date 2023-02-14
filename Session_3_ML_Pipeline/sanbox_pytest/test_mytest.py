from mytest import square

@pytest.fixture
def input_val():
    return 3


# def test_square_gives_correct_value():
#     subject=square(3)

#     assert subject == 9

def test_square_gives_correct_value(input_val):
    subject=square(input_val)

    assert subject == 9