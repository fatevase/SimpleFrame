""" 
    D: Dataset
    M: Model
    T: Trainer
    E: Evaluator
    L: Loading
    P: Preprocessing
    S: Saving
    R: Rebuiding
    eg: DP: dataset processing
"""
def processInfo(head, tail='', emoji=None):
    head_emoji_map = dict(
        D=':Hamburger:',
        M=':robot_face:',
        T=':rocket:',
        E=':microscope:',
        L=':cold_face:',
        P=':face_savoring_food:',
        S=':clapping_hands:',
        R=':face_with_symbols_on_mouth:'
    )
    if emoji is None:
        emoji = ""
        for h in list(head.split(' ')[0]):
                emoji += head_emoji_map[h]+' '

    return f"{emoji} {head} {tail}"
