from evaluate_openKBP import compute_metrics

if __name__ == '__main__':
    gt_dir = './dist/test_data'
    pred_path ='./dist/result/dose.nii.gz'

    Dose_score, DVH_score = compute_metrics(pred_path, gt_dir)
    print('Dose score is: ' + str(Dose_score))
    print('DVH score is: ' + str(DVH_score))