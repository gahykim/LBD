from scklearn.metrics import ndcg_score

def calculate_metric(output, target, min_rating, bin_size, k):
    indices = torch.argmax(output, dim = -1)
    y_pred = min_rating + indices * bin_size
    
    y_pred = y_pred.clone().detach().numpy()
    y_true = target.clone().detach().numpy()
    
    ndcg_score = ndcg_score(target, output, k)
    recall = len(set(y_pred) & set(y_true)) / len(target)

    return recall, ndcg_score
