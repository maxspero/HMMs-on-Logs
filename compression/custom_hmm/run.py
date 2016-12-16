from compression import *

if __name__ == "__main__":
    #train_model(100, 10, 10)
    #train_model(100, 20, 2, True)
    #train_model(100, 40, 2, True)
    results = [0] * 16
    for _ in xrange(5):
        results[0] += score_model('100-10')
        results[1] += score_model('100-10', topk=10)
        results[2] += score_model('100-10', topk=100)
        results[3] += score_model('100-10', topk=-1)
        results[4] += score_model('100-10-3')
        results[5] += score_model('100-10-3', topk=10)
        results[6] += score_model('100-10-3', topk=100)
        results[7] += score_model('100-10-3', topk=-1)
        results[8] += score_model('100-10-4')
        results[9] += score_model('100-10-4', topk=10)
        results[10] += score_model('100-10-4', topk=100)
        results[11] += score_model('100-10-4', topk=-1)
        results[12] += score_model('100-10-4')
        results[13] += score_model('100-10-10', topk=10)
        results[14] += score_model('100-10-10', topk=100)
        results[15] += score_model('100-10-10', topk=-1)
    results = [r/5 for r in results]
    print("Results: {}".format(results))

    """
    for subsequence_len in ['10', '20', '40']:
        results = [0] * 7
        for _ in xrange(5):
            model_name = '100-' + subsequence_len + '-2-triplets'
            results[0] += score_model(model_name)
            results[1] += score_model(model_name, topk=10)
            results[2] += score_model(model_name, topk=100)
            results[3] += score_model(model_name, topk=-1)
            results[4] += score_model(model_name, topk=10, triplets=True)
            results[5] += score_model(model_name, topk=100, triplets=True)
            results[6] += score_model(model_name, topk=-1, triplets=True)
        results = [r/5 for r in results]
        print("Results: {}".format(results))
    """
