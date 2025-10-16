# (Removed stray inserted lines.)
#!/usr/bin/env python3
import json
import math
import os
from math import log

import numpy as np
import textwrap
import argparse

# Attempt to load a local .env file (securely) if python-dotenv is installed.
# This allows users to place sensitive keys in an untracked .env file.
DOTENV_PATH_LOADED = None
try:
    from dotenv import load_dotenv, find_dotenv
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.isfile(env_path):
        # Do not override existing env vars; CLI flags should take precedence.
        load_dotenv(env_path, override=False)
        DOTENV_PATH_LOADED = env_path
    else:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
            DOTENV_PATH_LOADED = found
except Exception:
    DOTENV_PATH_LOADED = None

DATA_PATH = r'd:\Development\TheTower\GemAssessment\data.json'
OUTPUT_MD = r'd:\Development\TheTower\GemAssessment\analysis.md'


def parse_duration(s):
    s = s.strip()
    days = hours = minutes = seconds = 0.0
    # Split on spaces; tokens like '2d', '1h', '44m', '23s'
    for tok in s.split():
        tok = tok.strip()
        if tok.endswith('d'):
            try:
                days += float(tok[:-1])
            except:
                pass
        elif tok.endswith('h'):
            try:
                hours += float(tok[:-1])
            except:
                pass
        elif tok.endswith('m'):
            try:
                minutes += float(tok[:-1])
            except:
                pass
        elif tok.endswith('s'):
            try:
                seconds += float(tok[:-1])
            except:
                pass
    total_hours = days * 24.0 + hours + minutes / 60.0 + seconds / 3600.0
    return total_hours


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float('nan')
    return np.corrcoef(a, b)[0, 1]


with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

records = []
for rec in data:
    try:
        rem = parse_duration(rec['remaining_time'])
    except Exception:
        rem = float(rec.get('remaining_hours', float('nan')))
    try:
        init = parse_duration(rec['estimated_initial_duration'])
    except Exception:
        # if the string is already a number or malformed
        try:
            init = float(rec['estimated_initial_duration'])
        except Exception:
            init = float('nan')
    cost = float(rec['rush_cost'])
    per = cost / rem if rem > 0 else float('nan')
    records.append({'remaining_hours': rem, 'initial_hours': init, 'rush_cost': cost, 'per_hour': per})

rem_hours = np.array([r['remaining_hours'] for r in records])
init_hours = np.array([r['initial_hours'] for r in records])
per_hour = np.array([r['per_hour'] for r in records])
rush_costs = np.array([r['rush_cost'] for r in records])

# correlations
corr_per_rem = pearson(per_hour, rem_hours)
corr_per_init = pearson(per_hour, init_hours)
corr_rem_init = pearson(rem_hours, init_hours)

# Regression: per_hour vs ln(initial_hours)
valid_idx = np.isfinite(init_hours) & (init_hours > 0) & np.isfinite(per_hour)
Xln = np.log(init_hours[valid_idx])
y = per_hour[valid_idx]
coeffs_ln = np.polyfit(Xln, y, 1)
pred_ln = np.polyval(coeffs_ln, Xln)
resid_ln = y - pred_ln
ss_res_ln = np.sum(resid_ln ** 2)
ss_tot_ln = np.sum((y - y.mean()) ** 2)
r2_ln = 1 - ss_res_ln / ss_tot_ln if ss_tot_ln > 0 else float('nan')
rmse_ln = math.sqrt(np.mean(resid_ln ** 2))

# Regression: per_hour vs initial_hours (linear)
valid_idx_lin = np.isfinite(init_hours) & np.isfinite(per_hour)
X_lin = init_hours[valid_idx_lin]
y_lin = per_hour[valid_idx_lin]
coeffs_lin = np.polyfit(X_lin, y_lin, 1)
pred_lin = np.polyval(coeffs_lin, X_lin)
resid_lin = y_lin - pred_lin
ss_res_lin = np.sum(resid_lin ** 2)
ss_tot_lin = np.sum((y_lin - y_lin.mean()) ** 2)
r2_lin = 1 - ss_res_lin / ss_tot_lin if ss_tot_lin > 0 else float('nan')
rmse_lin = math.sqrt(np.mean(resid_lin ** 2))

# Multivariate regression: per_hour ~ intercept + init_hours + rem_hours
A = np.column_stack([np.ones_like(init_hours), init_hours, rem_hours])
mask = np.all(np.isfinite(A), axis=1) & np.isfinite(per_hour)
A2 = A[mask]
y2 = per_hour[mask]
try:
    beta_mv, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    pred_mv = A2.dot(beta_mv)
    resid_mv = y2 - pred_mv
    ss_res_mv = np.sum(resid_mv ** 2)
    ss_tot_mv = np.sum((y2 - y2.mean()) ** 2)
    r2_mv = 1 - ss_res_mv / ss_tot_mv if ss_tot_mv > 0 else float('nan')
    rmse_mv = math.sqrt(np.mean(resid_mv ** 2))
except Exception:
    beta_mv = [float('nan')] * 3
    r2_mv = float('nan')
    rmse_mv = float('nan')

# Regression: rush_cost ~ remaining_hours
mask_rc = np.isfinite(rem_hours) & np.isfinite(rush_costs)
coeffs_rc = np.polyfit(rem_hours[mask_rc], rush_costs[mask_rc], 1)
pred_rc = np.polyval(coeffs_rc, rem_hours[mask_rc])
resid_rc = rush_costs[mask_rc] - pred_rc
ss_res_rc = np.sum(resid_rc ** 2)
ss_tot_rc = np.sum((rush_costs[mask_rc] - rush_costs[mask_rc].mean()) ** 2)
r2_rc = 1 - ss_res_rc / ss_tot_rc if ss_tot_rc > 0 else float('nan')
rmse_rc = math.sqrt(np.mean(resid_rc ** 2))

# Full-model regression: cost = c + rem*(a + b*ln(T))
valid_cost_idx = np.isfinite(init_hours) & (init_hours > 0) & np.isfinite(rem_hours) & np.isfinite(rush_costs)
if np.any(valid_cost_idx):
    X_cost = np.column_stack([
        rem_hours[valid_cost_idx] * np.log(init_hours[valid_cost_idx]),
        rem_hours[valid_cost_idx],
        np.ones(np.count_nonzero(valid_cost_idx)),
    ])
    beta_cost, *_ = np.linalg.lstsq(X_cost, rush_costs[valid_cost_idx], rcond=None)
    b_cost = float(beta_cost[0])
    a_cost = float(beta_cost[1])
    c_cost = float(beta_cost[2])
    pred_cost = X_cost.dot(beta_cost)
    resid_cost = rush_costs[valid_cost_idx] - pred_cost
    ss_res_cost = np.sum(resid_cost ** 2)
    ss_tot_cost = np.sum((rush_costs[valid_cost_idx] - rush_costs[valid_cost_idx].mean()) ** 2)
    r2_cost = 1 - ss_res_cost / ss_tot_cost if ss_tot_cost > 0 else float('nan')
    rmse_cost = math.sqrt(np.mean(resid_cost ** 2))
else:
    a_cost = b_cost = c_cost = float('nan')
    r2_cost = rmse_cost = float('nan')

# Build markdown
lines = []
lines.append('# Lab Rush Cost Analysis')
lines.append('')
lines.append('Dataset: {} records'.format(len(records)))
lines.append('')
lines.append('## Key correlations')
lines.append('- Pearson corr(cost_per_hour, remaining_hours) = {:.4f}'.format(corr_per_rem))
lines.append('- Pearson corr(cost_per_hour, initial_hours) = {:.4f}'.format(corr_per_init))
lines.append('- Pearson corr(remaining_hours, initial_hours) = {:.4f}'.format(corr_rem_init))
lines.append('')
lines.append('## Regression: cost_per_hour vs ln(initial_hours)')
lines.append('Model: $\\mathrm{cost}_h = a + b \\cdot \\ln(T)$ where $T$ is initial duration in hours.')
lines.append('- a (intercept) = {:.4f}'.format(coeffs_ln[1]))
lines.append('- b (slope) = {:.4f}'.format(coeffs_ln[0]))
lines.append('- R^2 = {:.4f}'.format(r2_ln))
lines.append('- RMSE = {:.4f} gems/hour'.format(rmse_ln))
lines.append('')
lines.append('## Regression: cost_per_hour vs initial_hours (linear)')
lines.append('- intercept = {:.4f}, slope = {:.6f}, R^2 = {:.4f}, RMSE = {:.4f}'.format(coeffs_lin[1], coeffs_lin[0], r2_lin, rmse_lin))
lines.append('')
lines.append('## Multivariate regression: cost_per_hour ~ initial_hours + remaining_hours')
lines.append('- intercept = {:.4f}, beta_init = {:.6f}, beta_rem = {:.6f}'.format(beta_mv[0], beta_mv[1], beta_mv[2]))
lines.append('- R^2 = {:.4f}, RMSE = {:.4f}'.format(r2_mv, rmse_mv))
lines.append('')
lines.append('## rush_cost ~ remaining_hours')
lines.append('- slope = {:.4f}, intercept = {:.4f}, R^2 = {:.4f}, RMSE = {:.4f}'.format(coeffs_rc[0], coeffs_rc[1], r2_rc, rmse_rc))
lines.append('')
lines.append('## Empirical per-hour function')
lines.append(f"Using the ln(T) fit: r(T) ≈ {coeffs_ln[1]:.4f} + {coeffs_ln[0]:.4f} * ln(T) (T in hours).")
lines.append('')
lines.append('## Notes and recommendations')
lines.append('- The dominant predictor of gems/hour is initial total duration. Short jobs are charged more gems/hour.')
lines.append('- Remaining time correlates with gems/hour only because it is correlated with initial duration; its independent effect is small.')
lines.append('- The game likely uses a per-hour rate that depends on original duration, then multiplies by remaining hours and rounds to integer gems.')

with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# Parse runtime arguments: allow passing an API key without storing it and
# control whether default console output is shown. By default we suppress
# default response text; use --show-output to enable printing.
parser = argparse.ArgumentParser(description='Compute lab rush cost analysis')
parser.add_argument('-k', '--api-key', dest='api_key', help='OpenAI API key for this run (do not hard-code)')
parser.add_argument('--no-llm', dest='no_llm', action='store_true', help='Do not call an LLM; only compute and save analysis files')
parser.add_argument('--show-output', dest='show_output', action='store_true', help='Show analysis and LLM messages on console')
args = parser.parse_args()

# If an API key was supplied on the command line, set it for this process only.
if args.api_key:
    os.environ['OPENAI_API_KEY'] = args.api_key

# Informational messages about .env and API key location (do not print keys).
if args.show_output:
    if DOTENV_PATH_LOADED:
        print(f'Loaded environment variables from: {DOTENV_PATH_LOADED}')
    if args.api_key:
        print('OPENAI_API_KEY set for this run from command-line (value not shown).')
    elif os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY found in environment (from .env or external).')

# By default, do not print the full analysis text to console. Use --show-output
# to view the analysis interactively.
if args.show_output:
    print('\n'.join(lines))

# Build a compact analysis summary (JSON-serializable) to pass to an LLM
analysis_summary = {
    'n_records': len(records),
    'corr_per_rem': float(corr_per_rem),
    'corr_per_init': float(corr_per_init),
    'corr_rem_init': float(corr_rem_init),
    'ln_fit': {
        'intercept_a': float(coeffs_ln[1]),
        'slope_b': float(coeffs_ln[0]),
        'r2': float(r2_ln),
        'rmse': float(rmse_ln),
    },
    'linear_fit': {
        'intercept': float(coeffs_lin[1]),
        'slope': float(coeffs_lin[0]),
        'r2': float(r2_lin),
        'rmse': float(rmse_lin),
    },
    'multivariate': {
        'intercept': float(beta_mv[0]),
        'beta_init': float(beta_mv[1]),
        'beta_rem': float(beta_mv[2]),
        'r2': float(r2_mv),
        'rmse': float(rmse_mv),
    },
    'rush_vs_remaining': {
        'slope': float(coeffs_rc[0]),
        'intercept': float(coeffs_rc[1]),
        'r2': float(r2_rc),
        'rmse': float(rmse_rc),
    },
    'cost_model': {
        'a': float(a_cost),
        'b': float(b_cost),
        'c': float(c_cost),
        'r2': float(r2_cost),
        'rmse': float(rmse_cost),
    },
}


def build_llm_prompt(summary, examples=3):
    # Short, precise prompt for a concise technical write-up
    template = textwrap.dedent("""
    You are a concise technical data analyst. Produce a short (about 150-300 words) technical write-up
    addressed to a game analyst. Use only the numeric facts supplied; do not invent extra data.

    Tasks:
    1) Summarise the key findings in 3-6 sentences.
    2) Provide an explicit formula (LaTeX) for estimating the gem rush cost from initial total time T (hours)
       and remaining time rem (hours); include numeric coefficients derived from the data.
    3) State the model confidence, main caveats, and one practical recommendation for further data collection.

    JSON summary of computed analysis results:
    {json_summary}

    Provide the write-up as plain text. Keep it concise and factual.
    """).strip()
    json_summary = json.dumps(summary, indent=2)
    return template.replace('{json_summary}', json_summary)


def call_llm_with_prompt(prompt, model='gpt-5', api_key_env='OPENAI_API_KEY'):
    """Attempt to call an OpenAI ChatCompletion; return (text, error_string).
    If the OPENAI_API_KEY environment variable is not set, do not attempt a network call - return an explanatory error.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        return None, f"API key not set (env var {api_key_env}). Set it in your environment before calling the LLM."

    # Inspect installed openai package to determine which client API to call
    import importlib
    try:
        openai_mod = importlib.import_module('openai')
    except Exception as e:
        return None, f"openai python package not installed or failed to import: {e}. Install with `pip install openai`."

    version = getattr(openai_mod, '__version__', None)
    # Prefer new OpenAI client when available
    if hasattr(openai_mod, 'OpenAI'):
        try:
            client = openai_mod.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise technical data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=700,
            )
            choice0 = resp.choices[0]
            # message content can be in attributes or mapping
            try:
                text = choice0.message.content.strip()
            except Exception:
                try:
                    text = choice0['message']['content'].strip()
                except Exception:
                    text = str(choice0)
            return text, None
        except Exception as exc_new:
            msg = str(exc_new)
            # Common pattern for insufficient quota / 429 errors
            if 'insufficient_quota' in msg or '429' in msg or 'quota' in msg:
                suggestion = (
                    "Your OpenAI key or account appears to have insufficient quota (HTTP 429). "
                    "Check your OpenAI billing and usage or use a different API key. "
                    "You can also run this script with --no-llm to use the local fallback write-up."
                )
                return None, f"{msg}. SUGGESTION: {suggestion}"
            return None, f"New OpenAI client call failed (openai version={version}): {exc_new}"

    # If OpenAI class not present, try legacy (0.x) API if available
    if hasattr(openai_mod, 'ChatCompletion'):
        try:
            # set api key on module for legacy client
            try:
                openai_mod.api_key = api_key
            except Exception:
                pass
            resp = openai_mod.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise technical data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=700,
            )
            text = resp['choices'][0]['message']['content'].strip()
            return text, None
        except Exception as exc_old:
            # Compatibility issue: openai package may be >=1.0 and still expose ChatCompletion
            msg = str(exc_old)
            if 'insufficient_quota' in msg or '429' in msg or 'quota' in msg:
                suggestion = (
                    "Your OpenAI key or account appears to have insufficient quota (HTTP 429). "
                    "Check your OpenAI billing and usage or use a different API key. "
                    "You can also run this script with --no-llm to use the local fallback write-up."
                )
                return None, f"{msg}. SUGGESTION: {suggestion}"
            hint = (
                "The installed openai library does not support the legacy ChatCompletion API. "
                "If you have openai>=1.0, use the new OpenAI client or run 'openai migrate'. "
                "Alternatively, install the legacy client with 'pip install openai==0.28' (not recommended)."
            )
            return None, f"Legacy ChatCompletion call failed: {exc_old}. {hint}"

    # No compatible API surface found
    return None, (
        f"Installed openai package (version={version}) does not expose a compatible client. "
        "Install a supported openai package or set OPENAI_API_KEY."
    )


# Write the prompt to a file so the user can run it against any LLM if desired
PROMPT_PATH = os.path.splitext(OUTPUT_MD)[0] + '_prompt.txt'
with open(PROMPT_PATH, 'w', encoding='utf-8') as f:
    prompt = build_llm_prompt(analysis_summary)
    f.write(prompt)

if args.show_output:
    print('\nPrompt saved to: {}'.format(PROMPT_PATH))

# Try to call an LLM unless explicitly disabled. When output is suppressed
# we write errors to a file instead of printing them.
LLM_OUT = os.path.splitext(OUTPUT_MD)[0] + '_llm.md'
LLM_ERR_OUT = os.path.splitext(OUTPUT_MD)[0] + '_llm_error.txt'
if not args.no_llm:
    llm_text, llm_err = call_llm_with_prompt(prompt)
    if llm_text:
        with open(LLM_OUT, 'w', encoding='utf-8') as f:
            f.write(llm_text)
        if args.show_output:
            print('\nLLM write-up saved to: {}'.format(LLM_OUT))
            print('\n' + llm_text)
    else:
        # Save the error details so the run is auditable even when silent.
        with open(LLM_ERR_OUT, 'w', encoding='utf-8') as f:
            f.write(str(llm_err or 'LLM call failed with unknown error'))
        if args.show_output:
            print('\nLLM unavailable: {}'.format(llm_err))
            print('\nYou can run the prompt with any LLM by using the file: {}'.format(PROMPT_PATH))
else:
    if args.show_output:
        print('\nLLM call skipped (--no-llm). No write-up produced.')
