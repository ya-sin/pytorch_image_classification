def angle_convertion(x):
    return {
        0: 90,
        1: 45,
        2: 0,
        3: 315,
        4: 270,
        5: 225,
        6: 180,
        7: 135
    }[x]


def test():
    length = 3
    handwriting_result = ""
    result_num = []
    result_num.append(1)
    result_num.append(2)
    result_num.append(3)
    result_num.append(6)
    result_num.append(10)

    for i in range(0, length):
        handwriting_result = handwriting_result + str(result_num[i])
    print(handwriting_result)


if __name__ == "__main__":
    # angle = 1

    # print(angle_convertion(angle))
    test()
