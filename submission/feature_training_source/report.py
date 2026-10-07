"""Recompute saved predictions and make a transparent staged decision."""
from integration_common import *

def report():
    ensure_protocol(); d = arrays(); results = []
    ids_parts = []; predictions = {'control': [], 'augmented': []}
    for fold in range(5):
        out = OUT / f'fold{fold}'
        if not (out / 'result.json').exists():
            continue
        result = read(out / 'result.json'); ids = np.load(out / 'held_ids.npy'); yy = np.asarray(d['labels'][ids])
        ids_parts.append(ids); results.append(result)
        for arm in predictions:
            path = out / f'{arm}_probability.npy'; assert sha(path) == result['prediction_hashes'][path.name]
            p = np.load(path); assert abs(metric(yy, p)-result['scores'][arm]) < 1e-12
            predictions[arm].append(p)
    ids = np.concatenate(ids_parts); yy = np.asarray(d['labels'][ids]); incumbent = np.asarray(d['baseline'][ids])
    pp = {arm: np.concatenate(parts) for arm, parts in predictions.items()}
    scores = {arm: metric(yy, p) for arm, p in pp.items()}
    positive = sum(r['primary_delta'] > 0 for r in results); delta = scores['augmented']-scores['control']
    gate = len(results) == 5 and positive >= 4 and delta >= .001
    decision = 'failed_primary_gate' if not gate else 'requires_bootstrap'
    summary = {'completed_folds': len(results), 'series': len(ids), 'scores': scores, 'primary_delta': delta,
               'positive_folds': positive, 'incumbent': metric(yy, incumbent),
               'fixed_10pct_blends': {arm: metric(yy, .9*incumbent+.1*p) for arm,p in pp.items()},
               'decision': decision, 'primary_gate_pending_bootstrap': gate,
               'deployment_adopted': False, 'official_submission': False, 'official_score': False,
               'development_reused': True, 'independent_holdout': False, 'folds': results}
    write(OUT / 'result.json', summary)
    lines = ['# GPU特徴の直接統合：422列と566列の比較', '',
             f'完了：{len(results)}分割・{len(ids):,}系列。全対象予測から再計算したTS-AUCは **{scores["control"]:.9f} → {scores["augmented"]:.9f}（{delta:+.9f}）**。', '',
             '| Fold | 422特徴 | 566特徴 | 差 |', '|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["fold"]+1} | {r["scores"]["control"]:.9f} | {r["scores"]["augmented"]:.9f} | {r["primary_delta"]:+.9f} |')
    lines.extend(['', f'判定：{decision}。改善分割は{positive}/{len(results)}。', '',
                  '既存の因果的422特徴に、外側評価系列と特徴適用系列を学習から除外した予測器による144特徴を追加した。同じ対数間隔の学習時点・系列均等重み・3 seed×500 treesを使用。両条件とも4観測ごとに更新し、間は前値を保持して全観測を採点。', '',
                  '予測分布モデルの最初の2分割は既存チェックポイントを再利用。特徴キャッシュは生成し直し、系列IDとオンライン時点で厳密に結合した。評価系列では2つの予測器による特徴を別々に検知器へ入力し、予測確率を平均した。', '',
                  'これは反復利用した開発データの比較であり、独立テストでも公式Publicスコアでもない。全アンサンブルへの採用・公式提出は行っていない。'])
    (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'folds'}), flush=True)

if __name__ == '__main__':
    report()
