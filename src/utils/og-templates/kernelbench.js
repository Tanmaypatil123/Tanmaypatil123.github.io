import satori from "satori";
import loadGoogleFonts from "../loadGoogleFont";

// Share card for /kernelbench (1200x630), in the site's dark palette.
const BG = "#212737";
const FG = "#eaedf3";
const ACCENT = "#ff6b01";
const MUTED = "#8a95a8";

const title = "KernelBench-Hard × Hermes Agent";
const sub = "5 models · 6 GPU kernel problems · RTX PRO 6000";
const stats = [
  ["29/30", "kernels correct"],
  ["1–3%", "of peak on KDA & top-k"],
  ["9/30", "sessions read the grader"],
  ["0", "cheats found"],
];
const host = "tanmaypatil.com/kernelbench";

const div = (style, children) => ({ type: "div", props: { style: { display: "flex", ...style }, children } });

export default async () =>
  satori(
    div(
      { width: "100%", height: "100%", background: BG, color: FG, flexDirection: "column", padding: "64px 72px", justifyContent: "space-between" },
      [
        div({ flexDirection: "column" }, [
          div({ width: 96, height: 8, background: ACCENT, borderRadius: 4, marginBottom: 32 }, []),
          div({ fontSize: 52, fontWeight: 700, lineHeight: 1.15 }, title),
          div({ fontSize: 30, color: MUTED, marginTop: 20 }, sub),
        ]),
        div(
          { justifyContent: "space-between", gap: 20 },
          stats.map(([n, label]) =>
            div(
              { flexDirection: "column", flex: 1, border: `2px solid ${ACCENT}`, borderRadius: 12, padding: "20px 22px" },
              [div({ fontSize: 50, fontWeight: 700, color: ACCENT }, n), div({ fontSize: 21, color: FG, marginTop: 6 }, label)]
            )
          )
        ),
        div({ justifyContent: "space-between", fontSize: 24, color: MUTED }, [
          div({}, "All 30 kernels, harness and raw results"),
          div({ color: FG, fontWeight: 700 }, host),
        ]),
      ]
    ),
    {
      width: 1200,
      height: 630,
      embedFont: true,
      fonts: await loadGoogleFonts(title + sub + stats.flat().join("") + host + "All 30 kernels, harness and raw results"),
    }
  );
