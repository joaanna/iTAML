from collections import Counter
import numpy as np
import pickle
import glob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x) ** (1 / 2)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def get_correct(acc_task, task, chunks, start_point, class_per_task):
    correct = 0
    correct2 = 0
    task_scores = []
    class_scores = []
    targets = []
    targets_pred = []
    for t in range(task + 1):
        list_0 = []
        list_1 = []
        list_2 = []
        list_3 = []
        for i in range(chunks):
            acc_task_0 = acc_task[start_point + i]
            list_0.append(acc_task_0[t][0].detach().cpu().numpy())
            list_1.append(acc_task_0[t][1].detach().cpu().numpy())
            list_2.append(acc_task_0[t][2])
            list_3.append(acc_task_0[t][3].detach().cpu().numpy())
        list_0 = np.array(list_0)
        list_1 = np.array(list_1)
        list_2 = np.array(list_2)
        list_3 = np.array(list_3)

        targets_pred.append(list_0)
        class_scores.append(list_1)
        task_scores.append(list_2)
        targets.append(list_3)

    m = task_scores[0]
    task_scores2 = []
    for t2 in range(task + 1):
        m2 = m[:, t2:(t2 + 1)]
        m3 = np.max(m2, 1)
        task_scores2.append(np.mean(m3))
    pred_task = np.argmax(task_scores2)
    if (pred_task == targets[0][0] // class_per_task):
        correct2 += chunks
        for j in range(chunks):
            local_t = np.argmax(class_scores[pred_task][j])
            pred_x = [targets_pred[pred_task][j][local_t]]
            target_x = targets[0][j]
            if (target_x in pred_x + pred_task * class_per_task):
                correct += 1
            #         else:
    #             print(pred_task, targets[0][0]//class_per_task)
    return correct, correct2


def get_mata_score(p, task, chunks):
    with open(p + "/sample_per_task_testing_" + str(task) + ".pickle", 'rb') as handle:
        task_samples = pickle.load(handle)
    total_samples = np.sum([task_samples[x] for x in range(task + 1)])
    class_per_task = 10
    with open(p + "/meta_task_test_list_" + str(task) + ".pickle", 'rb') as handle:
        acc_task = pickle.load(handle)
    correct = 0
    correct2 = 0
    for tt in range(task + 1):
        ctask_samples = np.sum([task_samples[x] for x in range(tt)])
        for class_id in range(task_samples[tt] // chunks):
            start_point = ctask_samples + class_id * chunks
            c, c2 = get_correct(acc_task, task, chunks, start_point, class_per_task)
            correct += c
            correct2 += c2

        new_chunk = task_samples[tt] - (class_id + 1) * chunks
        if (new_chunk > 0):
            start_point = ((task_samples[tt] // chunks) * chunks)
            c, c2 = get_correct(acc_task, tt, new_chunk, start_point, class_per_task)
            correct += c
            correct2 += c2
    return correct / total_samples * 100, correct2 / total_samples * 100


CLASSES = ['1', '2', '4', '5', '10', '20', '25', '50']

for i in CLASSES:
    folder = 'models/cifar100/' + i + 'class_per_task'
    chunks = 20
    z = np.zeros_like(['']*int(i))
    z2 = np.zeros_like(['']*int(i))
    for j in range(10):
        z[j], z2[j] = get_mata_score(folder, j, chunks)
    print(folder, '\n', z)