from compression import *

if __name__ == "__main__":
    #train_model(100, 10)
    #train_model(100, 20)
    #train_model(100, 40)
    #train_model(1000, 10)
    #train_model(1000, 20)
    #train_model(1000, 40)
    results = [0] * 6
    for _ in xrange(5):
        results[0] += score_model('100-20')
        results[1] += score_model('100-20', topk=10)
        results[2] += score_model('100-20', topk=100)
        results[3] += score_model('100-20', topk=-1)
        results[4] += score_model('100-20', compressed=False)
        results[5] += score_model('100-20', compressed=False, modified=False)
    results = [r/5 for r in results]
    print("Results: {}".format(results))
