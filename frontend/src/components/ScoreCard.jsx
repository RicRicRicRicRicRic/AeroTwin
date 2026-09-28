import VulnerabilityBadge from './VulnerabilityBadge.jsx';

/**
 * ScoreCard — reusable seismic assessment result card.
 *
 * Renders the score, tier, weighted factor audit trail, aggregated defect
 * summary, and the rule-based recommendations produced by the backend scoring
 * engine. Handles pending / running / failed states too.
 */

const BAR_COLOURS = {
  age: 'bg-sky-500',
  construction: 'bg-indigo-500',
  stories: 'bg-violet-500',
  compliance: 'bg-fuchsia-500',
  crack_intensity: 'bg-rose-500',
  extent: 'bg-amber-500',
  element_damage: 'bg-red-600',
};

const FACTOR_LABELS = {
  age: 'Structure age',
  construction: 'Construction type',
  stories: 'Height (stories)',
  compliance: 'Code compliance',
  crack_intensity: 'Crack intensity',
  extent: 'Crack extent',
  element_damage: 'Element damage',
};

function Stat({ label, value, hint }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-sm font-semibold text-slate-800">{value}</div>
      {hint ? <div className="mt-0.5 text-[11px] text-slate-400">{hint}</div> : null}
    </div>
  );
}

export default function ScoreCard({ assessment, className = '' }) {
  if (!assessment) {
    return (
      <div
        className={`rounded-xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-500 ${className}`}
      >
        No assessment selected. Choose or run an assessment to see the vulnerability score.
      </div>
    );
  }

  if (assessment.status !== 'completed') {
    const failed = assessment.status === 'failed';
    return (
      <div
        className={`rounded-xl border p-6 text-sm ${
          failed
            ? 'border-red-200 bg-red-50 text-red-700'
            : 'border-slate-200 bg-white text-slate-600'
        } ${className}`}
      >
        <div className="font-semibold">
          {failed ? 'Assessment failed' : `Assessment ${assessment.status}…`}
        </div>
        <div className="mt-1 break-words text-xs">
          {failed ? assessment.error || 'Unknown error' : 'The scoring engine is running.'}
        </div>
      </div>
    );
  }

  const result = assessment.result;
  if (!result) {
    return (
      <div
        className={`rounded-xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-800 ${className}`}
      >
        Assessment completed but no result payload was stored.
      </div>
    );
  }

  const defect = result.defect_summary;
  const elementStates = defect.element_damage_states;

  return (
    <div
      className={`space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Building</div>
          <div className="text-lg font-semibold text-slate-800">{assessment.building_name}</div>
          <div className="mt-0.5 font-mono text-[11px] text-slate-400">
            assessment {String(assessment.assessment_id || '').slice(0, 12)}
          </div>
        </div>
        <VulnerabilityBadge
          classification={result.classification}
          score={result.vulnerability_score}
          size="lg"
          showScore
        />
      </div>

      <div>
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-bold text-slate-900">
            {result.vulnerability_score.toFixed(2)}
          </span>
          <span className="text-sm text-slate-500">/ 100 vulnerability index</span>
        </div>
        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-amber-500 to-red-600"
            style={{ width: `${Math.min(Math.max(result.vulnerability_score, 0), 100)}%` }}
          />
        </div>
      </div>

      <div>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Weighted factors
        </div>
        <div className="overflow-hidden rounded-lg border border-slate-200">
          <table className="min-w-full divide-y divide-slate-100 text-xs">
            <thead className="bg-slate-50 text-slate-500">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Factor</th>
                <th className="px-3 py-2 text-right font-medium">Value</th>
                <th className="px-3 py-2 text-right font-medium">Weight</th>
                <th className="px-3 py-2 text-right font-medium">Contribution</th>
                <th className="w-40 px-3 py-2 text-left font-medium">Share</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {result.factors.map((factor) => (
                <tr key={factor.name}>
                  <td className="px-3 py-1.5 text-slate-700">
                    {FACTOR_LABELS[factor.name] || factor.name}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-600">
                    {factor.value.toFixed(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-600">
                    {factor.weight.toFixed(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono font-semibold text-slate-800">
                    {factor.contribution.toFixed(4)}
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className={`h-full rounded-full ${
                          BAR_COLOURS[factor.name] || 'bg-slate-400'
                        }`}
                        style={{ width: `${Math.min(factor.contribution * 100, 100)}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Defect summary
        </div>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <Stat
            label="Frames analyzed"
            value={defect.frames_analyzed}
            hint={`${defect.frames_with_cracks} with cracks`}
          />
          <Stat label="Mean crack area" value={`${defect.mean_area_ratio.toFixed(4)} %`} />
          <Stat label="Mean crack width" value={`${defect.mean_width_px.toFixed(3)} px`} />
          <Stat label="Longest crack" value={`${defect.longest_crack_px.toFixed(1)} px`} />
        </div>
        {elementStates ? (
          <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-600">
            {Object.entries(elementStates).map(([state, count]) => (
              <span key={state} className="rounded-full bg-slate-100 px-2 py-0.5 font-medium">
                {state}: {count}
              </span>
            ))}
          </div>
        ) : (
          <div className="mt-2 text-[11px] text-slate-400">
            Element damage states unavailable — the element weight was redistributed across the
            remaining factors.
          </div>
        )}
      </div>

      <div>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Recommendations
        </div>
        <ul className="space-y-1 text-xs text-slate-700">
          {result.recommendations.map((recommendation) => (
            <li key={recommendation} className="flex gap-2">
              <span className="text-slate-400">•</span>
              <span>{recommendation}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
