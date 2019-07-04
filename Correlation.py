
def Correlation(samples):
    results={}
    for index in samples.testing_index:
        samples.Split(target=index)
        score = runLgb(samples.X, samples.y)
        results[index]=score
    print(results)