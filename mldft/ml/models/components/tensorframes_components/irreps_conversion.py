import regex as re
from e3nn.o3 import Irreps as E3nnIrreps

from tensorframes.reps import Irreps


def convert_to_tensorframes_irreps_string(e3nn_irreps_string: str) -> str:
    """Converts an E3NN irreps string to a TensorFrames irreps string.

    Args:
        e3nn_irreps_string (str): The E3NN irreps string to be converted.

    Returns:
        str: The converted TensorFrames irreps string.

    Raises:
        ValueError: If the input string is not in the correct format.

    Examples:
        >>> convert_to_tensorframes_irreps_string("2x3o+4x5o")
        '2x3n+4x5n'
    """

    # get rid of spaces:
    e3nn_irreps_string = e3nn_irreps_string.replace(" ", "")

    list_new_irreps = []

    # use regex to parse strings of that type 12x7e
    pattern = re.compile(r"(\d+)x(\d+)([eo])")

    for s in e3nn_irreps_string.split("+"):
        multiplier, l, even_odd = pattern.match(s).groups()
        if even_odd == "e" and int(l) % 2 == 0:
            # even and even -> normal tensor:
            list_new_irreps.append(f"{multiplier}x{l}n")
        elif even_odd == "o" and int(l) % 2 == 1:
            # odd and odd -> normal tensor:
            list_new_irreps.append(f"{multiplier}x{l}n")
        else:
            # pseudo tensor:
            list_new_irreps.append(f"{multiplier}x{l}p")

    return "+".join(list_new_irreps)


def convert_to_e3nn_irreps_string(tf_irreps_string: str) -> str:
    """Converts a TensorFlow irreps string to an E3NN irreps string.

    Args:
        tf_irreps_string (str): The TensorFlow irreps string to be converted.

    Returns:
        str: The converted E3NN irreps string.

    Raises:
        AttributeError: If the input string is not in the correct format.

    Example:
        >>> convert_to_e3nn_irreps_string("2x3n+4x5p")
        '2x3o+4x5e'
    """
    # get rid of spaces:
    tf_irreps_string = tf_irreps_string.replace(" ", "")

    list_new_irreps = []

    # use regex to parse strings of that type 12x7n
    pattern = re.compile(r"(\d+)x(\d+)([np])")

    for s in tf_irreps_string.split("+"):
        multiplier, l, pseudo_normal = pattern.match(s).groups()
        if pseudo_normal == "n":
            # normal tensor:
            even_odd = "e" if int(l) % 2 == 0 else "o"
            list_new_irreps.append(f"{multiplier}x{l}{even_odd}")
        else:
            # pseudo tensor:
            even_odd = "o" if int(l) % 2 == 0 else "e"
            list_new_irreps.append(f"{multiplier}x{l}{even_odd}")

    return "+".join(list_new_irreps)


def convert_e3nn_irreps_to_tensorframes(e3nn_irreps: E3nnIrreps) -> Irreps:
    """Converts E3nnIrreps to TensorFrames Irreps.

    Args:
        e3nn_irreps (E3nnIrreps): The E3nnIrreps object to be converted.

    Returns:
        Irreps: The converted Irreps object.
    """
    e3nn_irreps_string = str(e3nn_irreps)
    tf_irreps_string = convert_to_tensorframes_irreps_string(e3nn_irreps_string)
    return Irreps(tf_irreps_string)


def convert_tensorframes_irreps_to_e3nn(tf_irreps: Irreps) -> E3nnIrreps:
    """Converts TensorFrames irreps to E3nn irreps.

    Args:
        tf_irreps (Irreps): The TensorFrames irreps to be converted.

    Returns:
        E3nnIrreps: The converted E3nn irreps.
    """
    tf_irreps_string = str(tf_irreps)
    e3nn_irreps_string = convert_to_e3nn_irreps_string(tf_irreps_string)
    return E3nnIrreps(e3nn_irreps_string)


if __name__ == "__main__":
    e3nn_irreps = E3nnIrreps("12x7e+9x3o+5x4e+3x0e")
    print("E3NN irreps:", e3nn_irreps)
    tf_irreps = convert_e3nn_irreps_to_tensorframes(e3nn_irreps)
    print("TensorFrames irreps:", tf_irreps)
    e3nn_irreps_reconstructed = convert_tensorframes_irreps_to_e3nn(tf_irreps)
    print("cycle E3NN irreps:", e3nn_irreps_reconstructed)
    assert str(e3nn_irreps) == str(e3nn_irreps_reconstructed)
    print("Conversion successful!")
    tf_irreps_reconstructed = convert_e3nn_irreps_to_tensorframes(e3nn_irreps_reconstructed)
    assert str(tf_irreps) == str(tf_irreps_reconstructed)
    print("Second conversion successful too!")
