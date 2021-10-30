from proj1_helpers import *
from implementations import *

if __name__ == "__main__":
    #load training data
    y, tX, ids = load_csv_data('train.csv')

    # pre process training data
    xs_tr, ids_training = process_data(tX)
    ys_tr = y[ids_training[0]], y[ids_training[1]], y[ids_training[2]]

    # best model parameter found with CV
    model_params = [(12, 0.00010974987654930568), (12,0.002595024211399737), (12, 0.001023531021899027)]

    # create new model and compute the weights according to the best params and the training data
    predModel = ThreeModels(xs_tr, ys_tr)
    for i,m in enumerate(predModel.models):
        m.best_deg = model_params[i][0]
        m.best_lambda = model_params[i][1]
    predModel.compute_ws()
    
    # load test data
    _, tX_test, ids_test = load_csv_data('test.csv')

    # process test data
    xs_test, indexes_per_cat = process_data(tX_test)
    y_pred = np.zeros(len(tX_test))

    # make the prediction for each category
    for i, model in enumerate(predModel.models):
        x_poly = build_poly(xs_test[i], model.best_deg)
        y_pred[indexes_per_cat[i]] = predict_labels(model.w, x_poly)
    
    # save result
    create_csv_submission(ids_test, y_pred, 'prediction.csv')

