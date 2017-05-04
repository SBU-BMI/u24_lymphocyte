function bar_pred_truth_vals(truth, pred1, pred2)

bar([pred1, pred2, truth; 0, 0, 0]);
axis([0.7, 1.3, 0, 1.0]);
xlabel('Pred1   Pred2   Truth');
set(gca, 'XTickLabel', ['']);

