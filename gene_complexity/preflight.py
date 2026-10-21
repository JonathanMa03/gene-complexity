"""Validate the eight retained input files before running models."""
from gene_complexity._common import digest, read

HANDOFF = ('baron_scgpt_full.npz', 'baron_scgpt_full.metadata.json',
           'checkpoint_selection.json', 'preparation_report.json')
SPLIT_FILES = ('baron_split_manifest.csv', 'split_settings.json', 'baron_exclusions.csv')


def preflight(raw, splits, scgpt):
    paths = [raw, *[splits/n for n in SPLIT_FILES], *[scgpt/n for n in HANDOFF]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing retained inputs:\n' + '\n'.join(missing))
    hashes = {str(p): digest(p) for p in paths}
    settings = read(splits/'split_settings.json')
    if settings['status'] != 'final' or not settings['scgpt_confirmed']:
        raise ValueError('Retain the finalized, scGPT-confirmed split bundle')
    if hashes[str(raw)] != settings['input_sha256']:
        raise ValueError('Raw Baron hash differs from final split settings')
    if hashes[str(splits/'baron_split_manifest.csv')] != settings['manifest_sha256']:
        raise ValueError('Final manifest hash mismatch')
    meta, choice, prep = [read(scgpt/n) for n in HANDOFF[1:]]
    if meta['status'] != 'PASS' or hashes[str(scgpt/HANDOFF[0])] != meta['output_sha256']:
        raise ValueError('Invalid scGPT embedding metadata/hash')
    if choice['checkpoint_id'] != meta['model_id'] or choice['checkpoint_sha256'] != meta['checkpoint_sha256']:
        raise ValueError('scGPT checkpoint selection mismatch')
    if prep['status'] != 'PASS' or prep['output_sha256'] != meta['input_sha256'] or hashes[str(raw)] not in prep['input_hashes'].values():
        raise ValueError('scGPT preparation does not match shared raw data')
    return hashes


