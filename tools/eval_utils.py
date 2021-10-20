def accuracy(true, pred):
    acc = None
    right = 0
    wrong = 0
    for i in range(len(true)):
      if pred[i] == true[i]:
        right += 1
      else:
        wrong += 1
    acc = right / (right + wrong)
    return acc