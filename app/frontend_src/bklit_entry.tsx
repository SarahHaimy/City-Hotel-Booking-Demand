import React, { useMemo, useRef, useState, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { scaleLinear, scaleBand } from "d3-scale";
import { line as d3Line, curveMonotoneX } from "d3-shape";
import { LazyMotion, domAnimation, m } from "motion/react";

// Official @bklitui/ui chart primitives
import { BarChart } from "./ui-src/charts/bar-chart";
import { Bar } from "./ui-src/charts/bar";
import { BarXAxis } from "./ui-src/charts/bar-x-axis";
import { ComposedChart } from "./ui-src/charts/composed-chart";
import { SeriesBar } from "./ui-src/charts/series-bar";
import { Line } from "./ui-src/charts/line";
import { Grid } from "./ui-src/charts/grid";
import { XAxis } from "./ui-src/charts/x-axis";
import { YAxis } from "./ui-src/charts/y-axis";
import { ChartTooltip } from "./ui-src/charts/tooltip";
import { RadarChart } from "./ui-src/charts/radar-chart";
import { RadarGrid } from "./ui-src/charts/radar-grid";
import { RadarAxis } from "./ui-src/charts/radar-axis";
import { RadarLabels } from "./ui-src/charts/radar-labels";
import { RadarArea } from "./ui-src/charts/radar-area";
import { Gauge } from "./ui-src/charts/gauge";

const CREAM_PALETTE = [
  "#2F4550", // Smoked Slate (--chart-1)
  "#A57A3F", // Smoked Bronze (--chart-2)
  "#6F8A6A", // Muted Sage (--chart-3)
  "#B06A50", // Fired Clay (--chart-4)
  "#6E8BA0", // Smoked Blue Glass (--chart-5)
  "#8C7B6B", // Warm Taupe (--chart-6)
  "#7B5E73", // Dusk Plum-Grey
  "#C9AE7C", // Champagne Sand
];

function useBoxWidth(ref: React.RefObject<HTMLDivElement | null>, fallback = 640): number {
  const [w, setW] = useState<number>(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 40) setW(Math.floor(rect.width));
    };
    measure();
    const raf = requestAnimationFrame(measure);
    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(measure);
      ro.observe(el);
      return () => {
        cancelAnimationFrame(raf);
        ro.disconnect();
      };
    }
    return () => cancelAnimationFrame(raf);
  }, [ref]);
  return w;
}

function fmtNum(v: any, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(digits);
}

function LegendPills({ items }: { items?: { label: string; color: string }[] }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="bklit-legend-pills">
      {items.map((it, i) => (
        <span key={i} className="bklit-legend-item">
          <span className="bklit-legend-dot" style={{ background: it.color }} />
          <span>{it.label}</span>
        </span>
      ))}
    </div>
  );
}

/* 1. Official @bklitui/ui BarChart wrapper */
function BklitBarView({ spec }: { spec: any }) {
  const series: { key: string; label: string; color?: string }[] = spec.series || [];
  const legendItems = series.map((s, idx) => ({
    label: s.label || s.key,
    color: s.color || CREAM_PALETTE[idx % CREAM_PALETTE.length],
  }));
  const h = spec.height || 340;
  const formatRows = useMemo(
    () => (point: Record<string, unknown>) =>
      series
        .filter((s) => point[s.key] !== undefined && point[s.key] !== null)
        .map((s, idx) => ({
          color: s.color || CREAM_PALETTE[idx % CREAM_PALETTE.length],
          label: s.label || s.key,
          value: fmtNum(point[s.key], 4),
        })),
    [series]
  );

  return (
    <div>
      {legendItems.length > 1 && (
        <div style={{ marginBottom: 8 }}>
          <LegendPills items={legendItems} />
        </div>
      )}
      <div style={{ height: h, width: "100%", position: "relative" }}>
        <BarChart
          data={spec.data || []}
          xDataKey={spec.xKey || "category"}
          aspectRatio="auto"
          className="h-full w-full"
          margin={{ top: 18, right: 18, bottom: 42, left: 48 }}
          animationDuration={750}
        >
          <Grid horizontal />
          <YAxis />
          {series.map((s, idx) => {
            const col = s.color || CREAM_PALETTE[idx % CREAM_PALETTE.length];
            return (
              <Bar
                key={s.key}
                dataKey={s.key}
                fill={col}
                stroke={col}
                lineCap={4}
              />
            );
          })}
          <BarXAxis showAllLabels={(spec.data || []).length <= 16} />
          <ChartTooltip rows={formatRows} />
        </BarChart>
      </div>
      {spec.xTitle && (
        <div style={{ textAlign: "center", fontSize: 11, fontWeight: 600, color: "#55514A", marginTop: 2 }}>
          {spec.xTitle}
        </div>
      )}
    </div>
  );
}

/* 2. Official @bklitui/ui ComposedChart (Scree Bar + Cumulative Line) */
function BklitScreeView({ spec }: { spec: any }) {
  const baseMs = Date.UTC(2025, 0, 1);
  const dayMs = 86_400_000;
  const labels: string[] = (spec.data || []).map((r: any) => String(r.pc));
  const dataWithDates = useMemo(
    () =>
      (spec.data || []).map((r: any, idx: number) => ({
        ...r,
        date: new Date(baseMs + idx * dayMs),
      })),
    [spec.data]
  );
  const screeYDomains = useMemo(() => ({ left: [0, 1.05] as [number, number] }), []);
  const screeRows = useMemo(
    () => (point: Record<string, unknown>) => [
      {
        color: "#8C7B6B",
        label: "Individual Variance",
        value: `${(Number(point.individual || 0) * 100).toFixed(2)}%`,
      },
      {
        color: "#A57A3F",
        label: "Cumulative Variance",
        value: `${(Number(point.cumulative || 0) * 100).toFixed(2)}%`,
      },
    ],
    []
  );

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <LegendPills
          items={[
            { label: "Individual Variance", color: "#8C7B6B" },
            { label: "Cumulative Variance", color: "#A57A3F" },
          ]}
        />
      </div>
      <div style={{ height: spec.height || 340, width: "100%" }}>
        <ComposedChart
          data={dataWithDates}
          xDataKey="date"
          aspectRatio="auto"
          className="h-full w-full"
          margin={{ top: 18, right: 20, bottom: 40, left: 48 }}
          yDomains={screeYDomains}
          animationDuration={800}
        >
          <Grid horizontal />
          <YAxis />
          <SeriesBar dataKey="individual" fill="#8C7B6B" radius={4} />
          <Line dataKey="cumulative" stroke="#A57A3F" strokeWidth={2.5} showMarkers fadeEdges={false} />
          <XAxis labels={labels} />
          <ChartTooltip rows={screeRows} showDatePill={false} />
        </ComposedChart>
      </div>
      {spec.subtitleNote && (
        <div style={{ fontSize: 11.5, color: "#6E695F", marginTop: 6 }}>{spec.subtitleNote}</div>
      )}
    </div>
  );
}

/* 3. Official @bklitui/ui RadarChart */
function BklitRadarView({ spec }: { spec: any }) {
  const rows: any[] = spec.data || [];
  const metrics: { key: string; label: string }[] = spec.metrics || [];
  const legendItems = rows.map((r, i) => ({
    label: r.label || r.id,
    color: CREAM_PALETTE[i % CREAM_PALETTE.length],
  }));
  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <LegendPills items={legendItems} />
      </div>
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: spec.height || 360 }}>
        <RadarChart data={rows} metrics={metrics} size={330} margin={58} enterDurationMs={800}>
          <RadarGrid />
          <RadarAxis />
          <RadarLabels fontSize={11} offset={18} />
          {rows.map((r, i) => {
            const col = CREAM_PALETTE[i % CREAM_PALETTE.length];
            return <RadarArea key={r.id || i} index={i} color={col} />;
          })}
        </RadarChart>
      </div>
    </div>
  );
}

/* 4. Official @bklitui/ui Gauge */
function BklitGaugeView({ spec }: { spec: any }) {
  const pct = Math.max(0, Math.min(100, Number(spec.value || 0)));
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: spec.height || 210 }}>
      <div style={{ width: "100%", maxWidth: 300 }}>
        <Gauge
          value={pct}
          centerValue={Number(pct.toFixed(1))}
          totalNotches={38}
          defaultLabel={spec.unitLabel || "Cancel Risk %"}
        />
      </div>
    </div>
  );
}

/* 5. Diverging Horizontal Signed Bars (Feature Attributions / Coefficients) */
function SignedHorizontalBarView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 580);
  const [tip, setTip] = useState<{ x: number; y: number; label: string; val: number } | null>(null);
  const rows: { feature: string; value: number }[] = spec.data || [];
  const rowH = 24;
  const margin = { top: 10, right: 26, bottom: 32, left: 190 };
  const height = Math.max(spec.height || 340, rows.length * rowH + margin.top + margin.bottom);
  const innerW = Math.max(120, width - margin.left - margin.right);
  const innerH = height - margin.top - margin.bottom;

  const maxAbs = Math.max(1e-6, ...rows.map((r) => Math.abs(Number(r.value) || 0))) * 1.08;
  const hasNeg = rows.some((r) => Number(r.value) < 0);
  const xScale = scaleLinear().domain([hasNeg ? -maxAbs : 0, maxAbs]).range([0, innerW]).nice();
  const yScale = scaleBand<string>()
    .domain(rows.map((r) => r.feature))
    .range([0, innerH])
    .padding(0.28);
  const zeroX = xScale(0);
  const ticks = xScale.ticks(5);

  return (
    <div ref={boxRef} style={{ position: "relative", width: "100%" }}>
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {ticks.map((t, i) => (
            <g key={i} transform={`translate(${xScale(t)},0)`}>
              <line y1={0} y2={innerH} stroke="rgba(58, 55, 51, 0.11)" strokeDasharray="3 3" />
              <text y={innerH + 18} textAnchor="middle" className="bklit-tick">
                {fmtNum(t, 2)}
              </text>
            </g>
          ))}
          <line x1={zeroX} x2={zeroX} y1={0} y2={innerH} stroke="#C6A874" strokeWidth={1.3} />
          {rows.map((r, idx) => {
            const v = Number(r.value) || 0;
            const y = yScale(r.feature) ?? 0;
            const bw = Math.max(2, Math.abs(xScale(v) - zeroX));
            const x = v >= 0 ? zeroX : xScale(v);
            const fill = v >= 0 ? "#2F4550" : "#B06A50";
            const shortLbl = r.feature.length > 26 ? r.feature.slice(0, 25) + "…" : r.feature;
            return (
              <g
                key={r.feature + idx}
                onMouseEnter={(e) => {
                  const rect = boxRef.current?.getBoundingClientRect();
                  setTip({
                    x: (e.clientX - (rect?.left ?? 0)) + 12,
                    y: (e.clientY - (rect?.top ?? 0)) - 10,
                    label: r.feature,
                    value: v,
                  } as any);
                }}
                onMouseLeave={() => setTip(null)}
              >
                <text
                  x={-10}
                  y={y + yScale.bandwidth() / 2}
                  dy="0.35em"
                  textAnchor="end"
                  style={{ fontSize: 11, fill: "#3A3733", fontFamily: "'IBM Plex Mono', monospace" }}
                >
                  {shortLbl}
                </text>
                <m.rect
                  initial={{ width: 0, x: zeroX }}
                  animate={{ width: bw, x }}
                  transition={{ duration: 0.45, delay: idx * 0.015 }}
                  y={y}
                  height={yScale.bandwidth()}
                  rx={4}
                  fill={fill}
                  opacity={0.9}
                />
              </g>
            );
          })}
        </g>
      </svg>
      {tip && (
        <div className="bklit-float-tooltip" style={{ left: Math.min(tip.x, width - 190), top: Math.max(0, tip.y) }}>
          <div style={{ fontWeight: 600 }}>{tip.label}</div>
          <div style={{ color: "#D8C094" }}>Attribution: {fmtNum((tip as any).value, 4)}</div>
        </div>
      )}
    </div>
  );
}

/* 6. Continuous Multi-Series XY Curves (ROC, PR, Calibration, Threshold Sweep, k-Selection, What-If) */
function ContinuousCurvesView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 560);
  const height = spec.height || 350;
  const margin = { top: 14, right: 24, bottom: 42, left: 52 };
  const innerW = Math.max(120, width - margin.left - margin.right);
  const innerH = Math.max(120, height - margin.top - margin.bottom);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const lines: { name: string; x: number[]; y: number[]; color?: string; dashed?: boolean }[] = spec.lines || [];
  const allX = lines.flatMap((l) => l.x);
  const allY = lines.flatMap((l) => l.y);
  const xMin = spec.xDomain ? spec.xDomain[0] : Math.min(0, ...allX);
  const xMax = spec.xDomain ? spec.xDomain[1] : Math.max(1, ...allX);
  const yMin = spec.yDomain ? spec.yDomain[0] : Math.min(0, ...allY);
  const yMax = spec.yDomain ? spec.yDomain[1] : Math.max(1, ...allY) * 1.05;

  const xScale = scaleLinear().domain([xMin, xMax]).range([0, innerW]);
  const yScale = scaleLinear().domain([yMin, yMax]).range([innerH, 0]).nice();

  const pathGen = d3Line<[number, number]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]))
    .curve(curveMonotoneX);

  const legendItems = lines.map((l, i) => ({
    label: l.name,
    color: l.color || (l.dashed ? "#A89F90" : CREAM_PALETTE[i % CREAM_PALETTE.length]),
  }));

  // Nearest point per curve at hoverX
  const hoverRows = useMemo(() => {
    if (hoverX === null) return null;
    const targetX = xScale.invert(hoverX);
    return {
      targetX,
      items: lines
        .filter((l) => !l.dashed && l.x.length > 0)
        .map((l, idx) => {
          let bestIdx = 0;
          let bestDist = Infinity;
          for (let i = 0; i < l.x.length; i++) {
            const dist = Math.abs(l.x[i] - targetX);
            if (dist < bestDist) {
              bestDist = dist;
              bestIdx = i;
            }
          }
          return {
            name: l.name,
            color: l.color || CREAM_PALETTE[idx % CREAM_PALETTE.length],
            y: l.y[bestIdx],
            x: l.x[bestIdx],
          };
        }),
    };
  }, [hoverX, lines, innerW]);

  return (
    <div ref={boxRef} style={{ position: "relative", width: "100%" }}>
      <div style={{ marginBottom: 8 }}>
        <LegendPills items={legendItems} />
      </div>
      <svg
        width={width}
        height={height}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const lx = e.clientX - rect.left - margin.left;
          if (lx >= 0 && lx <= innerW) setHoverX(lx);
          else setHoverX(null);
        }}
        onMouseLeave={() => setHoverX(null)}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {yScale.ticks(5).map((t, i) => (
            <g key={i} transform={`translate(0,${yScale(t)})`}>
              <line x1={0} x2={innerW} stroke="rgba(58, 55, 51, 0.11)" strokeDasharray="3 3" />
              <text x={-8} dy="0.32em" textAnchor="end" className="bklit-tick">
                {fmtNum(t, 2)}
              </text>
            </g>
          ))}
          {xScale.ticks(6).map((t, i) => (
            <g key={i} transform={`translate(${xScale(t)},${innerH})`}>
              <text y={18} textAnchor="middle" className="bklit-tick">
                {fmtNum(t, 2)}
              </text>
            </g>
          ))}
          {spec.hline !== undefined && (
            <line
              x1={0}
              x2={innerW}
              y1={yScale(spec.hline)}
              y2={yScale(spec.hline)}
              stroke="#A89F90"
              strokeDasharray="4 4"
            />
          )}
          {spec.vline !== undefined && (
            <g transform={`translate(${xScale(spec.vline)},0)`}>
              <line y1={0} y2={innerH} stroke="#A57A3F" strokeDasharray="4 3" strokeWidth={1.6} />
              {spec.vlineLabel && (
                <text x={5} y={13} style={{ fontSize: 10.5, fill: "#8A6A3B", fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace" }}>
                  {spec.vlineLabel}
                </text>
              )}
            </g>
          )}
          {lines.map((l, i) => {
            const pts: [number, number][] = l.x.map((xv, k) => [xv, l.y[k]]);
            const d = pathGen(pts) || "";
            const col = l.color || (l.dashed ? "#A89F90" : CREAM_PALETTE[i % CREAM_PALETTE.length]);
            return (
              <g key={l.name + i}>
                <m.path
                  d={d}
                  fill="none"
                  stroke={col}
                  strokeWidth={l.dashed ? 1.4 : 2.3}
                  strokeDasharray={l.dashed ? "5 4" : undefined}
                  initial={{ pathLength: 0, opacity: 0.4 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.65 }}
                />
                {!l.dashed && l.x.length <= 24 &&
                  pts.map((pt, k) => (
                    <circle key={k} cx={xScale(pt[0])} cy={yScale(pt[1])} r={3.3} fill={col} stroke="#FBF9F5" strokeWidth={1.1} />
                  ))}
              </g>
            );
          })}
          {hoverX !== null && (
            <line x1={hoverX} x2={hoverX} y1={0} y2={innerH} stroke="#A57A3F" strokeWidth={1.1} strokeDasharray="2 2" />
          )}
          {spec.xTitle && (
            <text x={innerW / 2} y={innerH + 35} textAnchor="middle" className="bklit-axis-title">
              {spec.xTitle}
            </text>
          )}
          {spec.yTitle && (
            <text
              transform={`translate(-38,${innerH / 2}) rotate(-90)`}
              textAnchor="middle"
              className="bklit-axis-title"
            >
              {spec.yTitle}
            </text>
          )}
        </g>
      </svg>
      {hoverRows && hoverRows.items.length > 0 && hoverX !== null && (
        <div
          className="bklit-float-tooltip"
          style={{ left: Math.min(hoverX + margin.left + 12, width - 210), top: 38 }}
        >
          <div style={{ color: "#D8C094", marginBottom: 3 }}>
            {spec.xTitle || "x"}: {fmtNum(hoverRows.targetX, 3)}
          </div>
          {hoverRows.items.map((it, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span className="bklit-legend-dot" style={{ background: it.color }} />
              <span>{it.name}:</span>
              <strong style={{ marginLeft: "auto", paddingLeft: 8 }}>{fmtNum(it.y, 4)}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* 7. Continuous 2D Scatter (Parity, Residuals, 2D PCA Clusters, Pareto Time vs Score) */
function ContinuousScatterView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 560);
  const height = spec.height || 370;
  const margin = { top: 16, right: 24, bottom: 42, left: 54 };
  const innerW = Math.max(120, width - margin.left - margin.right);
  const innerH = Math.max(120, height - margin.top - margin.bottom);
  const [tip, setTip] = useState<{ x: number; y: number; text: string; group: string } | null>(null);

  const groups: { name: string; x: number[]; y: number[]; labels?: string[]; color?: string; size?: number }[] =
    spec.groups || [];
  const allX = groups.flatMap((g) => g.x);
  const allY = groups.flatMap((g) => g.y);
  const xScale = scaleLinear()
    .domain([Math.min(...allX, 0), Math.max(...allX, 1) * 1.03])
    .range([0, innerW])
    .nice();
  const yScale = scaleLinear()
    .domain([Math.min(...allY, spec.includeZeroY ? 0 : Infinity), Math.max(...allY, 1) * 1.04])
    .range([innerH, 0])
    .nice();

  const legendItems = groups.map((g, i) => ({
    label: g.name,
    color: g.color || CREAM_PALETTE[i % CREAM_PALETTE.length],
  }));

  return (
    <div ref={boxRef} style={{ position: "relative", width: "100%" }}>
      {legendItems.length > 1 && (
        <div style={{ marginBottom: 8 }}>
          <LegendPills items={legendItems} />
        </div>
      )}
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {yScale.ticks(5).map((t, i) => (
            <g key={i} transform={`translate(0,${yScale(t)})`}>
              <line x1={0} x2={innerW} stroke="rgba(58, 55, 51, 0.11)" strokeDasharray="3 3" />
              <text x={-8} dy="0.32em" textAnchor="end" className="bklit-tick">
                {fmtNum(t, 1)}
              </text>
            </g>
          ))}
          {xScale.ticks(6).map((t, i) => (
            <g key={i} transform={`translate(${xScale(t)},${innerH})`}>
              <line y1={-innerH} y2={0} stroke="rgba(58, 55, 51, 0.06)" />
              <text y={18} textAnchor="middle" className="bklit-tick">
                {fmtNum(t, 1)}
              </text>
            </g>
          ))}
          {spec.diagonal && (
            <line
              x1={xScale(0)}
              y1={yScale(0)}
              x2={xScale(Math.min(xScale.domain()[1], yScale.domain()[1]))}
              y2={yScale(Math.min(xScale.domain()[1], yScale.domain()[1]))}
              stroke="#A57A3F"
              strokeWidth={1.6}
              strokeDasharray="5 4"
            />
          )}
          {spec.zeroHline && (
            <line x1={0} x2={innerW} y1={yScale(0)} y2={yScale(0)} stroke="#23211E" strokeWidth={1.3} />
          )}
          {groups.map((g, gi) => {
            const col = g.color || CREAM_PALETTE[gi % CREAM_PALETTE.length];
            const r = g.size || (g.x.length > 500 ? 2.6 : 6.5);
            const alpha = g.x.length > 500 ? 0.44 : 0.88;
            return (
              <g key={g.name + gi}>
                {g.x.map((xv, i) => {
                  const cx = xScale(xv);
                  const cy = yScale(g.y[i]);
                  const lbl = g.labels?.[i];
                  return (
                    <g key={i}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={r}
                        fill={col}
                        fillOpacity={alpha}
                        onMouseEnter={() =>
                          setTip({
                            x: cx + margin.left + 10,
                            y: cy + margin.top - 10,
                            group: g.name,
                            text: lbl || `(${fmtNum(xv, 2)}, ${fmtNum(g.y[i], 2)})`,
                          })
                        }
                        onMouseLeave={() => setTip(null)}
                      />
                      {spec.showLabels && lbl && (
                        <text x={cx} y={cy - 9} textAnchor="middle" style={{ fontSize: 10.5, fill: "#3A3733", fontFamily: "'Manrope', sans-serif" }}>
                          {lbl}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}
          {spec.xTitle && (
            <text x={innerW / 2} y={innerH + 35} textAnchor="middle" className="bklit-axis-title">
              {spec.xTitle}
            </text>
          )}
          {spec.yTitle && (
            <text
              transform={`translate(-40,${innerH / 2}) rotate(-90)`}
              textAnchor="middle"
              className="bklit-axis-title"
            >
              {spec.yTitle}
            </text>
          )}
        </g>
      </svg>
      {tip && (
        <div className="bklit-float-tooltip" style={{ left: Math.min(tip.x, width - 220), top: Math.max(4, tip.y) }}>
          <div style={{ fontWeight: 600 }}>{tip.group}</div>
          <div style={{ color: "#D8C094" }}>{tip.text}</div>
        </div>
      )}
    </div>
  );
}

/* 8. Matrix Heatmap (Confusion Matrix, Cluster Profile Heatmap, PCA Loadings) */
function MatrixHeatmapView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 520);
  const [tip, setTip] = useState<{ x: number; y: number; row: string; col: string; z: number; txt?: string } | null>(null);
  const xLabels: string[] = spec.xLabels || [];
  const yLabels: string[] = spec.yLabels || [];
  const zMat: number[][] = spec.z || [];
  const textMat: string[][] | undefined = spec.text;
  const diverging = Boolean(spec.diverging);

  const cellH = Math.max(26, Math.min(58, Math.floor(260 / Math.max(1, yLabels.length))));
  const margin = { top: 12, right: 20, bottom: 48, left: yLabels.some((s) => s.length > 16) ? 175 : 120 };
  const height = Math.max(spec.height || 300, yLabels.length * cellH + margin.top + margin.bottom);
  const innerW = Math.max(120, width - margin.left - margin.right);
  const innerH = height - margin.top - margin.bottom;

  const flat = zMat.flat();
  const minZ = Math.min(...flat, 0);
  const maxZ = Math.max(...flat, 1e-6);
  const absMax = Math.max(Math.abs(minZ), Math.abs(maxZ), 1e-6);

  const cellColor = (val: number): { bg: string; fg: string } => {
    if (diverging) {
      const t = Math.max(-1, Math.min(1, val / absMax));
      if (t >= 0) {
        const a = 0.10 + 0.78 * t;
        return { bg: `rgba(165, 122, 63, ${a.toFixed(3)})`, fg: t > 0.52 ? "#FAF5EC" : "#23211E" };
      }
      const a = 0.10 + 0.76 * Math.abs(t);
      return { bg: `rgba(47, 69, 80, ${a.toFixed(3)})`, fg: Math.abs(t) > 0.50 ? "#FAF5EC" : "#23211E" };
    }
    const t = maxZ > minZ ? (val - minZ) / (maxZ - minZ) : 0;
    const a = 0.10 + 0.80 * t;
    return { bg: `rgba(47, 69, 80, ${a.toFixed(3)})`, fg: t > 0.50 ? "#FAF5EC" : "#23211E" };
  };

  const xBand = scaleBand<string>().domain(xLabels).range([0, innerW]).padding(0.08);
  const yBand = scaleBand<string>().domain(yLabels).range([0, innerH]).padding(0.08);

  return (
    <div ref={boxRef} style={{ position: "relative", width: "100%" }}>
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {yLabels.map((yl, r) =>
            xLabels.map((xl, c) => {
              const val = zMat[r]?.[c] ?? 0;
              const { bg, fg } = cellColor(val);
              const cx = xBand(xl) ?? 0;
              const cy = yBand(yl) ?? 0;
              const rawTxt = textMat?.[r]?.[c];
              const lines = rawTxt ? String(rawTxt).split(/<br\s*\/?>|\n/) : [];
              return (
                <g
                  key={`${r}-${c}`}
                  onMouseEnter={() =>
                    setTip({
                      x: cx + margin.left + xBand.bandwidth() / 2,
                      y: cy + margin.top,
                      row: yl,
                      col: xl,
                      z: val,
                      txt: lines.join(" "),
                    })
                  }
                  onMouseLeave={() => setTip(null)}
                >
                  <rect
                    x={cx}
                    y={cy}
                    width={xBand.bandwidth()}
                    height={yBand.bandwidth()}
                    rx={6}
                    fill={bg}
                    stroke="rgba(165, 132, 79, 0.16)"
                    strokeWidth={0.75}
                  />
                  {lines.length > 0 &&
                    lines.map((ln, li) => (
                      <text
                        key={li}
                        x={cx + xBand.bandwidth() / 2}
                        y={cy + yBand.bandwidth() / 2 + (li - (lines.length - 1) / 2) * 14}
                        dy="0.35em"
                        textAnchor="middle"
                        style={{
                          fontSize: li === 0 ? 12 : 10.5,
                          fontWeight: li === 0 ? 600 : 400,
                          fontFamily: "'IBM Plex Mono', monospace",
                          fill: fg,
                        }}
                      >
                        {ln}
                      </text>
                    ))}
                </g>
              );
            })
          )}
          {yLabels.map((yl, r) => (
            <text
              key={r}
              x={-10}
              y={(yBand(yl) ?? 0) + yBand.bandwidth() / 2}
              dy="0.35em"
              textAnchor="end"
              style={{ fontSize: 11, fill: "#3A3733", fontFamily: "'Manrope', sans-serif" }}
            >
              {yl.length > 24 ? yl.slice(0, 23) + "…" : yl}
            </text>
          ))}
          {xLabels.map((xl, c) => (
            <text
              key={c}
              x={(xBand(xl) ?? 0) + xBand.bandwidth() / 2}
              y={innerH + 18}
              textAnchor="middle"
              style={{ fontSize: 11, fill: "#3A3733", fontFamily: "'Manrope', sans-serif" }}
            >
              {xl.length > 14 ? xl.slice(0, 13) + "…" : xl}
            </text>
          ))}
        </g>
      </svg>
      {tip && (
        <div className="bklit-float-tooltip" style={{ left: Math.min(tip.x, width - 210), top: Math.max(0, tip.y - 28) }}>
          <div style={{ fontWeight: 600 }}>{tip.row} · {tip.col}</div>
          <div style={{ color: "#D8C094" }}>{tip.txt ? `${tip.txt} (z=${fmtNum(tip.z, 2)})` : fmtNum(tip.z, 3)}</div>
        </div>
      )}
    </div>
  );
}

/* 9. Baseline -> Tuned Dumbbell Chart */
function DumbbellView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 560);
  const [tip, setTip] = useState<{ x: number; y: number; model: string; base: number; tuned: number } | null>(null);
  const rows: { model: string; baseline: number; tuned: number }[] = spec.data || [];
  if (rows.length === 0) {
    return <div style={{ padding: 18, fontSize: 12.5, color: "#6E695F" }}>Run both baseline and tuned variants to display optimization gains.</div>;
  }
  const margin = { top: 14, right: 32, bottom: 34, left: 175 };
  const height = Math.max(220, rows.length * 56 + margin.top + margin.bottom);
  const innerW = Math.max(140, width - margin.left - margin.right);
  const innerH = height - margin.top - margin.bottom;

  const vals = rows.flatMap((r) => [r.baseline, r.tuned]);
  const pad = (Math.max(...vals) - Math.min(...vals) || 1) * 0.18;
  const xScale = scaleLinear()
    .domain([Math.min(...vals) - pad, Math.max(...vals) + pad])
    .range([0, innerW])
    .nice();
  const yBand = scaleBand<string>().domain(rows.map((r) => r.model)).range([0, innerH]).padding(0.4);

  return (
    <div ref={boxRef} style={{ position: "relative", width: "100%" }}>
      <div style={{ marginBottom: 8 }}>
        <LegendPills
          items={[
            { label: "Baseline (Untuned)", color: "#B9AE9C" },
            { label: "Selected / Tuned", color: "#A57A3F" },
          ]}
        />
      </div>
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {xScale.ticks(5).map((t, i) => (
            <g key={i} transform={`translate(${xScale(t)},0)`}>
              <line y1={0} y2={innerH} stroke="rgba(58, 55, 51, 0.11)" strokeDasharray="3 3" />
              <text y={innerH + 20} textAnchor="middle" className="bklit-tick">
                {fmtNum(t, 3)}
              </text>
            </g>
          ))}
          {rows.map((r, idx) => {
            const cy = (yBand(r.model) ?? 0) + yBand.bandwidth() / 2;
            const x1 = xScale(r.baseline);
            const x2 = xScale(r.tuned);
            return (
              <g
                key={r.model + idx}
                style={{ cursor: "pointer" }}
                onMouseEnter={() =>
                  setTip({
                    x: margin.left + (x1 + x2) / 2,
                    y: margin.top + cy - 18,
                    model: r.model,
                    base: r.baseline,
                    tuned: r.tuned,
                  })
                }
                onMouseLeave={() => setTip(null)}
              >
                <text x={-12} y={cy} dy="0.35em" textAnchor="end" style={{ fontSize: 12, fontWeight: 600, fill: "#23211E" }}>
                  {r.model}
                </text>
                <line x1={x1} x2={x2} y1={cy} y2={cy} stroke="#D9C29A" strokeWidth={3.5} strokeLinecap="round" />
                <circle cx={x1} cy={cy} r={6} fill="#B9AE9C" stroke="#FBF9F5" strokeWidth={1.5} />
                <m.circle
                  initial={{ cx: x1 }}
                  animate={{ cx: x2 }}
                  transition={{ duration: 0.55 }}
                  cy={cy}
                  r={7}
                  fill="#A57A3F"
                  stroke="#FBF9F5"
                  strokeWidth={1.5}
                />
                <text x={x2 + (x2 >= x1 ? 12 : -12)} y={cy - 11} textAnchor="middle" className="bklit-tick">
                  {fmtNum(r.tuned, 3)}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      {tip && (
        <div className="bklit-float-tooltip" style={{ left: Math.min(tip.x, width - 220), top: Math.max(4, tip.y) }}>
          <div style={{ fontWeight: 600 }}>{tip.model}</div>
          <div style={{ color: "#D8C094" }}>
            Baseline: <strong style={{ color: "#FAF5EC" }}>{fmtNum(tip.base, 4)}</strong> → Selected:{" "}
            <strong style={{ color: "#E8D9BE" }}>{fmtNum(tip.tuned, 4)}</strong> ({tip.tuned - tip.base >= 0 ? "+" : ""}
            {fmtNum(tip.tuned - tip.base, 4)})
          </div>
        </div>
      )}
    </div>
  );
}

/* 10. Boxplot (CV Fold Scores & Hyperparameter Slice) */
function BoxplotView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 560);
  const height = spec.height || 330;
  const groups: { label: string; values: number[] }[] = spec.groups || [];
  const margin = { top: 16, right: 24, bottom: 44, left: 54 };
  const innerW = Math.max(120, width - margin.left - margin.right);
  const innerH = Math.max(120, height - margin.top - margin.bottom);

  const allV = groups.flatMap((g) => g.values);
  const pad = (Math.max(...allV, 1) - Math.min(...allV, 0) || 0.1) * 0.15;
  const yScale = scaleLinear()
    .domain([Math.min(...allV) - pad, Math.max(...allV) + pad])
    .range([innerH, 0])
    .nice();
  const xBand = scaleBand<string>().domain(groups.map((g) => g.label)).range([0, innerW]).padding(0.45);

  const quantile = (sorted: number[], q: number) => {
    if (sorted.length === 0) return 0;
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
  };

  return (
    <div ref={boxRef}>
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {yScale.ticks(5).map((t, i) => (
            <g key={i} transform={`translate(0,${yScale(t)})`}>
              <line x1={0} x2={innerW} stroke="rgba(58, 55, 51, 0.11)" strokeDasharray="3 3" />
              <text x={-8} dy="0.32em" textAnchor="end" className="bklit-tick">
                {fmtNum(t, 3)}
              </text>
            </g>
          ))}
          {groups.map((g, gi) => {
            const s = [...g.values].sort((a, b) => a - b);
            if (s.length === 0) return null;
            const q1 = quantile(s, 0.25);
            const med = quantile(s, 0.5);
            const q3 = quantile(s, 0.75);
            const lo = s[0];
            const hi = s[s.length - 1];
            const bx = xBand(g.label) ?? 0;
            const bw = Math.min(56, xBand.bandwidth());
            const cx = bx + xBand.bandwidth() / 2;
            const col = CREAM_PALETTE[gi % CREAM_PALETTE.length];
            return (
              <g key={g.label + gi}>
                <line x1={cx} x2={cx} y1={yScale(lo)} y2={yScale(hi)} stroke={col} strokeWidth={1.5} />
                <line x1={cx - bw * 0.3} x2={cx + bw * 0.3} y1={yScale(lo)} y2={yScale(lo)} stroke={col} strokeWidth={1.5} />
                <line x1={cx - bw * 0.3} x2={cx + bw * 0.3} y1={yScale(hi)} y2={yScale(hi)} stroke={col} strokeWidth={1.5} />
                <rect
                  x={cx - bw / 2}
                  y={yScale(q3)}
                  width={bw}
                  height={Math.max(2, yScale(q1) - yScale(q3))}
                  rx={4}
                  fill={col}
                  fillOpacity={0.20}
                  stroke={col}
                  strokeWidth={1.6}
                />
                <line x1={cx - bw / 2} x2={cx + bw / 2} y1={yScale(med)} y2={yScale(med)} stroke={col} strokeWidth={2.4} />
                {s.map((v, k) => (
                  <circle
                    key={k}
                    cx={cx + ((k % 3) - 1) * 5}
                    cy={yScale(v)}
                    r={3.2}
                    fill={col}
                    stroke="#FBF9F5"
                    strokeWidth={1}
                  />
                ))}
                <text x={cx} y={innerH + 18} textAnchor="middle" style={{ fontSize: 11, fill: "#3A3733", fontFamily: "'Manrope', sans-serif" }}>
                  {g.label.length > 18 ? g.label.slice(0, 17) + "…" : g.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

/* 11. Parallel Coordinates & Dendrogram */
function ParcoordsView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 600);
  const height = spec.height || 340;
  const dims: { label: string; values: number[]; tickvals?: number[]; ticktext?: string[] }[] = spec.dimensions || [];
  const margin = { top: 26, right: 36, bottom: 24, left: 36 };
  const innerW = Math.max(160, width - margin.left - margin.right);
  const innerH = Math.max(140, height - margin.top - margin.bottom);
  if (dims.length < 2) return null;

  const scales = dims.map((d) => {
    const mn = Math.min(...d.values);
    const mx = Math.max(...d.values);
    return scaleLinear().domain([mn, mx === mn ? mn + 1 : mx]).range([innerH, 0]);
  });
  const xStep = innerW / (dims.length - 1);
  const nRows = dims[0]?.values.length ?? 0;
  const scoreDim = dims[dims.length - 1];
  const sMin = Math.min(...scoreDim.values);
  const sMax = Math.max(...scoreDim.values);

  return (
    <div ref={boxRef}>
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {Array.from({ length: nRows }).map((_, r) => {
            const pts = dims.map((d, di) => `${di * xStep},${scales[di](d.values[r])}`).join(" ");
            const t = sMax > sMin ? (scoreDim.values[r] - sMin) / (sMax - sMin) : 0.5;
            const stroke = t > 0.66 ? "#A57A3F" : t > 0.33 ? "#2F4550" : "#B9AE9C";
            return <polyline key={r} points={pts} fill="none" stroke={stroke} strokeWidth={2} strokeOpacity={0.72} />;
          })}
          {dims.map((d, di) => (
            <g key={di} transform={`translate(${di * xStep},0)`}>
              <line y1={0} y2={innerH} stroke="#C9BFAE" strokeWidth={1.3} />
              <text y={-10} textAnchor="middle" style={{ fontSize: 11, fontWeight: 600, fill: "#23211E" }}>
                {d.label}
              </text>
              <text y={innerH + 15} textAnchor="middle" className="bklit-tick">
                {d.ticktext ? d.ticktext[0] : fmtNum(scales[di].domain()[0], 2)}
              </text>
              <text y={10} x={6} className="bklit-tick">
                {d.ticktext ? d.ticktext[d.ticktext.length - 1] : fmtNum(scales[di].domain()[1], 2)}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

function DendrogramView({ spec }: { spec: any }) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const width = useBoxWidth(boxRef, 580);
  const height = spec.height || 340;
  const icoord: number[][] = spec.icoord || [];
  const dcoord: number[][] = spec.dcoord || [];
  const margin = { top: 14, right: 20, bottom: 34, left: 50 };
  const innerW = Math.max(140, width - margin.left - margin.right);
  const innerH = Math.max(140, height - margin.top - margin.bottom);
  const allX = icoord.flat();
  const allY = dcoord.flat();
  const xScale = scaleLinear().domain([Math.min(...allX, 0), Math.max(...allX, 1)]).range([0, innerW]);
  const yScale = scaleLinear().domain([0, Math.max(...allY, 1) * 1.05]).range([innerH, 0]);

  return (
    <div ref={boxRef}>
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {icoord.map((xs, i) => {
            const ys = dcoord[i];
            const pts = xs.map((xv, k) => `${xScale(xv)},${yScale(ys[k])}`).join(" ");
            return <polyline key={i} points={pts} fill="none" stroke="#2F4550" strokeWidth={1.6} />;
          })}
          <text x={innerW / 2} y={innerH + 24} textAnchor="middle" className="bklit-axis-title">
            Merged Subtrees (Representative Sample)
          </text>
        </g>
      </svg>
    </div>
  );
}

function BklitChartShell({ spec }: { spec: any }) {
  if (!spec || typeof spec !== "object") return null;
  const kind = spec.kind;

  return (
    <LazyMotion features={domAnimation}>
      <div className="bklit-cream-root bklit-card">
        {(spec.title || spec.subtitle) && (
          <div className="bklit-header">
            <div>
              {spec.title && <div className="bklit-title">{spec.title}</div>}
              {spec.subtitle && <div className="bklit-subtitle">{spec.subtitle}</div>}
            </div>
          </div>
        )}
        {kind === "bklit_bar" && <BklitBarView spec={spec} />}
        {kind === "bklit_scree" && <BklitScreeView spec={spec} />}
        {kind === "bklit_radar" && <BklitRadarView spec={spec} />}
        {kind === "bklit_gauge" && <BklitGaugeView spec={spec} />}
        {kind === "signed_hbar" && <SignedHorizontalBarView spec={spec} />}
        {kind === "xy_curves" && <ContinuousCurvesView spec={spec} />}
        {kind === "xy_scatter" && <ContinuousScatterView spec={spec} />}
        {kind === "matrix_heatmap" && <MatrixHeatmapView spec={spec} />}
        {kind === "dumbbell" && <DumbbellView spec={spec} />}
        {kind === "boxplot" && <BoxplotView spec={spec} />}
        {kind === "parcoords" && <ParcoordsView spec={spec} />}
        {kind === "dendrogram" && <DendrogramView spec={spec} />}
      </div>
    </LazyMotion>
  );
}

/* Streamlit v2 custom component mount entrypoint */
const ROOTS = new WeakMap<Node, { container: HTMLDivElement; root: Root }>();

export default function mountStreamlitBklit(component: {
  data: any;
  parentElement: HTMLElement | ShadowRoot;
}) {
  const { data, parentElement } = component;
  let entry = ROOTS.get(parentElement);
  if (!entry) {
    const container = document.createElement("div");
    container.style.width = "100%";
    parentElement.appendChild(container);
    const root = createRoot(container);
    entry = { container, root };
    ROOTS.set(parentElement, entry);
  }
  entry.root.render(<BklitChartShell spec={data} />);
  return () => {
    // Allow clean unmount if Streamlit tears down the ShadowRoot
  };
}
