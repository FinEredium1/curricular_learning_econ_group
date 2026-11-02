"""Audit saved team outputs and build housing and cross-group reports.

Run from the research folder: .venv/bin/python reports/build_team_report.py
This script never trains or changes frozen scientific inputs.
"""
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

MODELS = ['linear', 'mlp', 'patchtst']
ORDERS = ['random', 'auxiliary_first', 'target_first', 'interleaved']
GROUPS = {1: 'Output and income', 2: 'Labor market', 3: 'Housing', 6: 'Interest and exchange rates'}
TEAM = ROOT / 'reports/team'
HOUSING = ROOT / 'reports/housing'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    """Require all conditions, matching provenance, and independently reproduced errors."""
    metadata = read(ROOT / 'data/prepared/metadata.json')
    manifest = read(ROOT / 'artifacts/batches_seed1.json')
    freeze = read(ROOT / 'artifacts/freeze.json')
    for relative, expected in freeze['files'].items():
        assert digest(ROOT / relative) == expected, relative
    for group in GROUPS:
        block = manifest['groups'][str(group)]
        expected = Counter(manifest['target_300'] + block['auxiliary'])
        for order in ORDERS:
            assert Counter(block['orders'][order]) == expected, (group, order)
        cycle = ['T', 'A', 'A', 'T', 'A', 'A', 'T', 'A'] * 100
        t, a = iter(manifest['target_300']), iter(block['auxiliary'])
        assert block['orders']['interleaved'] == [next(t) if slot == 'T' else next(a) for slot in cycle]
    assert len(manifest['groups']['3']['auxiliary']) == 500
    records, runs, checks = [], {}, []
    reference = None
    paths = list((ROOT / 'results/study').glob('group*/*/*/seed1/summary.json'))
    paths += list((ROOT.parent / 'group6_results_seed1_20261031T203006Z_154216/group6').glob('*/*/seed1/summary.json'))
    for path in sorted(paths):
        summary = read(path)
        group, model, order = summary['group'], summary['model'], summary['order']
        assert not summary['smoke_only'] and summary['seed'] == 1
        assert group in GROUPS and model in MODELS
        budget = 300 if order == 'target_only_300' else 800
        assert summary['updates'] == budget
        assert summary['snapshot_sha256'] == metadata['snapshot_sha256']
        assert summary['freeze_sha256'] == digest(ROOT / 'artifacts/freeze.json')
        assert summary['initial_checkpoint_sha256'] == digest(ROOT / f'artifacts/initial_{model}_seed1.pt')
        directory = path.parent
        curve = pd.read_csv(directory / 'validation.csv')
        assert curve['update'].tolist() == [*range(30, budget, 30), budget]
        assert np.isfinite(curve.select_dtypes(include='number')).all().all()
        predictions = pd.read_csv(directory / 'predictions.csv')
        assert len(predictions) == 354
        assert set(predictions.horizon) == {1, 2, 3}
        assert not predictions.duplicated(['forecast_origin', 'horizon']).any()
        assert np.isfinite(predictions[['actual', 'prediction']]).all().all()
        identity = predictions[['forecast_origin', 'forecast_month', 'horizon', 'actual']]
        if reference is None:
            reference = identity
        else:
            pd.testing.assert_frame_equal(reference, identity, check_exact=True)
        residual = predictions.prediction - predictions.actual
        for key, value in [('test_mse', np.mean(residual**2)), ('test_mae', np.mean(abs(residual))),
                           ('zero_growth_mse', np.mean(predictions.actual**2)),
                           ('zero_growth_mae', np.mean(abs(predictions.actual)))]:
            assert np.isclose(summary[key], value, rtol=1e-12, atol=1e-14), (path, key)
        checkpoint_path = directory / 'final.pt'
        checkpoint_verified = checkpoint_path.exists()
        if checkpoint_verified:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['step'] == budget
            assert {int(s['step'].item()) for s in checkpoint['optim_state_dict']['state'].values()} == {budget}
            assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
        if group == 3:
            assert checkpoint_verified, path
        key = (0 if order.startswith('target_only') else group, model, order)
        assert key not in runs, key
        runs[key] = (directory, curve, predictions)
        records.append({**summary, 'group_name': GROUPS[group], 'source_path': str(directory.relative_to(ROOT.parent)),
                        'checkpoint_verified': checkpoint_verified})
        checks.append({'path': str(path.relative_to(ROOT.parent)), 'summary_sha256': digest(path),
                       'predictions_sha256': digest(directory/'predictions.csv'),
                       'checkpoint_verified': checkpoint_verified})
    expected = {(g, m, o) for g in GROUPS for m in MODELS for o in ORDERS}
    expected |= {(0, m, o) for m in MODELS for o in ['target_only_300', 'target_only_800']}
    assert set(runs) == expected, expected - set(runs)
    assert len(records) == 54
    table = pd.DataFrame(records)
    for budget in (300, 800):
        controls = table[table.order == f'target_only_{budget}'].set_index('model').test_mse
        table[f'mse_improvement_vs_target{budget}_pct'] = [100*(1-row.test_mse/controls[row.model]) for row in table.itertuples()]
    return metadata, manifest, table, runs, checks


def markdown_table(table, columns):
    """Avoid optional tabulate dependency."""
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---']*len(columns)) + '|']
    for _, row in table.iterrows():
        lines.append('| ' + ' | '.join(f'{row[c]:.6f}' if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)


def figures(metadata, manifest, runs, table):
    """Reuse the team's existing figure conventions for housing."""
    spec = importlib.util.spec_from_file_location('team_figures', ROOT/'reports/build_teammate2_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORTS = HOUSING
    module.ORDERS = ORDERS + ['target_only_800', 'target_only_300']
    module.LABELS['target_only_300'] = 'INDPRO only (300)'
    module.COLORS['target_only_300'] = '#111111'
    housing_runs = {(m, o): runs[(0 if o.startswith('target_only') else 3, m, o)] for m in MODELS for o in module.ORDERS}
    module.plot_curves(housing_runs)
    module.plot_forecasts(metadata, housing_runs)
    # Schedule plotting uses four orders and group 1 internally; supply housing's actual manifests.
    module.ORDERS = ORDERS + ['target_only_800']
    module.plot_schedules({'target_300': manifest['target_300'], 'groups': {'1': manifest['groups']['3']}})
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for axis, model in zip(axes, MODELS):
        subset = table[(table.model == model) & (table.order == 'random')].set_index('group')
        axis.bar(range(4), [subset.loc[g, 'test_mse'] for g in GROUPS], color=['#2878b5','#e1812c','#3a923a','#9467bd'])
        for order, style in [('target_only_300', ':'), ('target_only_800', '--')]:
            value = table[(table.model == model) & (table.order == order)].iloc[0].test_mse
            axis.axhline(value, color='gray', linestyle=style, label=order)
        axis.axhline(table.iloc[0].zero_growth_mse, color='black', label='Zero growth')
        axis.set_xticks(range(4), ['Output/income','Labor','Housing','Rates/FX'], rotation=25, ha='right')
        axis.set(title=model, ylabel='INDPRO test MSE')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=3)
    figure.savefig(TEAM/'group_selection.png', dpi=170)
    plt.close(figure)


def write_reports(metadata, manifest, table, runs, checks):
    table.to_csv(TEAM/'results.csv', index=False)
    housing = table[table.group == 3]
    housing.to_csv(HOUSING/'results.csv', index=False)
    transitions = []
    for model in MODELS:
        for order, before, after in [('auxiliary_first',480,510),('target_first',300,330)]:
            curve = runs[(3,model,order)][1].set_index('update')
            transitions.append({'model':model,'order':order,'before_update':before,'before_mse':curve.loc[before,'mse_growth_units'],
                                'after_update':after,'after_mse':curve.loc[after,'mse_growth_units'],
                                'final_mse':curve.loc[800,'mse_growth_units']})
    transitions = pd.DataFrame(transitions)
    transitions.to_csv(HOUSING/'stage_transitions.csv', index=False)
    common = [
        'September 2026 FRED-MD; 24 monthly inputs predict three months. Auxiliary examples predict their own series. Training/validation/test forecast months fall in 1960–1999 / 2000–2009 / 2010–2019. Each series uses training-only scaling (ddof=0).',
        'Each augmented condition uses the same 300 INDPRO batches and 500 auxiliary batches, batch size 32. Linear, MLP, and PatchTST start from their common saved weights with fresh Adam optimizers (learning rate 0.001). Seed 1, CPU, float32, one thread, no dropout or early stopping. Final weights are evaluated after exactly 800 updates; controls use 300 or 800.',
        'MSE and MAE average all 118 forecast origins × three horizons after reversing INDPRO standardization. Zero growth means zero in original growth units.',
        'All 54 saved summaries and prediction files passed provenance, validation-cadence, finite-value, identical-target, and direct MSE/MAE checks. Batch identity/multiplicity and the exact interleaving cycle were checked for every group. Available final checkpoints were checked against optimizer step counters. Labor checkpoints were not committed, so their optimizer counters cannot be independently inspected here.',
        'Hardware differs across teammates; training times are retained but are not pooled or ranked. Local hardware and installed versions are recorded in ../housing_execution/. Python 3.11.18 is used locally rather than the original 3.11.7 patch release; all pinned packages and frozen scientific files verify unchanged.',
        'One seed provides preliminary results without uncertainty estimates or significance claims. Revised historical data do not establish real-time forecasting performance. These comparisons do not establish a general ranking of categories. Test results describe the prespecified conditions and were not used to tune or retrain.',
    ]
    zero = table.iloc[0]
    lines = ['# Housing study — Teammate 3', '', '## Question and method', '',
             'Does complete-group housing training help INDPRO forecasting, and does changing the order of identical batches affect the result?', '',
             common[0], '', common[1], '',
             'Housing includes all ten series: '+', '.join(metadata['groups']['3'])+'. Each receives exactly 50 auxiliary batches. Shared frozen inputs were reused; preparation was not rerun.', '',
             '## Final test results', '', common[2], '',
             f'Zero-growth MSE: {zero.zero_growth_mse:.6f}; MAE: {zero.zero_growth_mae:.6f}.', '',
             markdown_table(housing, ['model','order','test_mse','test_mae','mse_improvement_vs_target300_pct','mse_improvement_vs_target800_pct']), '',
             'Positive improvement percentages indicate lower MSE than the named shared control. Controls are in the combined team results.', '', '## Findings', '']
    for model in MODELS:
        subset = housing[housing.model == model]
        best = subset.loc[subset.test_mse.idxmin()]
        random = subset[subset.order == 'random'].iloc[0]
        lines += [f'- {model}: lowest housing test MSE is {best.test_mse:.6f} with {best.order}. Random mixing changes MSE by {random.mse_improvement_vs_target300_pct:.2f}% improvement versus the 300-update control and {random.mse_improvement_vs_target800_pct:.2f}% versus the 800-update control.']
    lines += ['', 'Random mixing improves MSE over both trained controls for all three models. Against zero growth, MLP random mixing improves MSE by about 0.88% and MAE by about 0.05%; MLP interleaving also narrowly improves MSE. The other housing runs do not beat zero growth on MSE. These small differences should not be treated as reliable gains from one seed.', '', 'Auxiliary-first has the highest housing test MSE for every model. The best schedule differs by model, so housing does not support a universal ordering advantage.', '', '## Training-stage transitions', '',
              'Auxiliary-first switches after update 500: the saved checkpoints at 480 and 510 bracket that switch. Target-first switches after 300: compare 300 and 330. These measurements also include training between checkpoints, so they do not isolate an instantaneous causal effect.', '',
              markdown_table(transitions,list(transitions.columns)), '', 'At the auxiliary-to-target transition, validation MSE initially falls for all models, but MLP and PatchTST finish with higher validation error than at update 480. After target-to-housing switching, all three models initially improve from update 300 to 330. The later curves fluctuate, especially for PatchTST; these observations do not isolate a mechanism.', '',
              '## Verification and limitations', '', *[p+'\n' for p in common[3:]],
              'Preparation time is unavailable because frozen artifacts were reused. Window counts are in ../../data/prepared/window_counts.csv. Figures show validation curves, batch sources, and the same four prespecified evenly spaced test windows. Full outputs and final checkpoints are under ../../results/study/group3/.', '']
    (HOUSING/'REPORT.md').write_text('\n'.join(lines))
    selection = table[table.order == 'random'].sort_values(['model','test_mse'])
    controls = table[table.order.str.startswith('target_only')].sort_values(['model','order'])
    lines = ['# Team experiment: economic groups and training order', '',
             '**Complete: 48 augmented runs plus six shared controls, seed 1.**', '',
             '## Method', '', common[0], '', common[1], '', common[2], '',
             '## Group selection under random mixing', '',
             markdown_table(selection,['model','group_name','test_mse','test_mae','mse_improvement_vs_target300_pct','mse_improvement_vs_target800_pct']), '',
             'Positive percentages mean lower MSE than the indicated control.', '',
             '## Shared target-only controls', '', markdown_table(controls,['model','order','test_mse','test_mae']), '',
             f'Zero growth: MSE {zero.zero_growth_mse:.6f}; MAE {zero.zero_growth_mae:.6f}.', '',
             '## Findings', '']
    for model in MODELS:
        subset = selection[selection.model == model]
        best = subset.iloc[0]
        lines.append(f'- Under random mixing, {best.group_name.lower()} has the lowest observed test MSE for {model}: {best.test_mse:.6f}. This is a descriptive single-seed comparison.')
    lines += ['', 'Every random-mixing group/model condition improves test MSE over both trained target-only controls in this seed. Labor random mixing also beats zero growth on both MSE and MAE for all three models. This comparison controls optimizer updates with the 800-update baseline and target exposure with the 300-update baseline.', '', 'The best observed schedule varies by group and model. Target-first has the lowest MSE for all three labor models; housing favors interleaving for Linear/PatchTST and random mixing for MLP. Auxiliary-first is the highest-MSE schedule for every housing, labor, and output/income model. Rates/FX PatchTST is an exception: random mixing is worse than the other schedules. There is no universal winning order.', '', '## Ordering within each group', '']
    for group, name in GROUPS.items():
        lines += [f'### {name}', '', markdown_table(table[(table.group == group) & table.order.isin(ORDERS)],['model','order','test_mse','test_mae']), '']
    lines += ['## Verification and limits', '', *[p+'\n' for p in common[3:]],
              f'Snapshot SHA-256: {metadata["snapshot_sha256"]}', '', f'Freeze SHA-256: {digest(ROOT/"artifacts/freeze.json")}', '',
              'Source branches: output-and-income-jayden (group 1 and 800 controls), labor-market-lisa (group 2 and 300 controls), Fardin_group6_results (group 6), plus locally completed housing runs. Imported files were preserved. Original teammate reports may describe comparisons as pending; this report supersedes those pending-status notes.', '',
              'See results.csv for all 54 runs, verification.json for the audit, group_selection.png for comparisons, and ../housing/REPORT.md for the housing study.', '']
    (TEAM/'REPORT.md').write_text('\n'.join(lines))
    status = {'runs':len(table),'augmented_runs':48,'shared_controls':6,'housing_runs':12,
              'all_saved_metrics_reproduced':True,'identical_test_targets':True,'shared_provenance':True,
              'checkpoint_counters_verified':int(table.checkpoint_verified.sum()),
              'missing_checkpoints':table.loc[~table.checkpoint_verified,'source_path'].tolist(),
              'frozen_files_unchanged':True,'platform':platform.platform(),'python':platform.python_version(),'files':checks}
    (TEAM/'verification.json').write_text(json.dumps(status,indent=2)+'\n')


def main():
    TEAM.mkdir(parents=True,exist_ok=True)
    HOUSING.mkdir(parents=True,exist_ok=True)
    metadata, manifest, table, runs, checks = collect()
    figures(metadata, manifest, runs, table)
    write_reports(metadata, manifest, table, runs, checks)
    print(table[table.group == 3][['model','order','test_mse','test_mae']].to_string(index=False))
    print('Verified 54 runs; housing and team reports written.')


if __name__ == '__main__':
    main()
