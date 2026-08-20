import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  PieChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

type Hit = {
  task: string;
  target: string;
  source: string;
  feature: string;
  n: number | null;
  rho: number;
  q: number;
};

const HITS: Hit[] = [
  { task: "誕生日パーティ", target: "CSS", source: "examiner", feature: "win_examiner_turn_rate_delta", n: null, rho: 0.295884, q: 0.021978 },
  { task: "誕生日パーティ", target: "SA", source: "examiner", feature: "win_examiner_turn_rate_delta", n: null, rho: 0.330254, q: 0.026973 },
  { task: "誕生日パーティ", target: "SA", source: "examiner", feature: "examiner_backchannel_frac", n: null, rho: -0.292159, q: 0.035964 },
  { task: "誕生日パーティ", target: "SA", source: "examiner", feature: "examiner_pitch_reversal_rate", n: null, rho: -0.288354, q: 0.035964 },
  { task: "おやつ", target: "RRB", source: "examiner", feature: "examiner_roll_range", n: null, rho: -0.316534, q: 0.046753 },
  { task: "おやつ", target: "RRB", source: "dyad", feature: "rsp_after_exam_gap_med", n: null, rho: 0.276316, q: 0.046753 },
  { task: "おやつ", target: "RRB", source: "child", feature: "txt_child_utt_char_cv", n: null, rho: -0.272025, q: 0.046753 },
  { task: "おやつ", target: "RRB", source: "examiner", feature: "examiner_sway_y", n: null, rho: -0.279011, q: 0.046753 },
  { task: "おやつ", target: "RRB", source: "child", feature: "child_sway_y", n: null, rho: -0.279335, q: 0.046753 },
  { task: "おやつ", target: "SA", source: "child", feature: "txt_child_question_frac", n: null, rho: 0.391675, q: 0.01998 },
  { task: "おやつ", target: "SA", source: "child", feature: "rsp_child_reply_frac", n: null, rho: -0.318721, q: 0.033966 },
  { task: "おやつ", target: "SA", source: "dyad", feature: "rsp_after_exam_gap_med", n: null, rho: 0.285729, q: 0.04662 },
  { task: "ものを用いたルーティンの期待反応", target: "CSS", source: "dyad", feature: "rsp_after_exam_gap_med", n: null, rho: -0.336501, q: 0.031968 },
  { task: "ものを用いたルーティンの期待反応", target: "CSS", source: "child", feature: "child_backchannel_frac", n: null, rho: -0.332633, q: 0.031968 },
  { task: "ものを用いたルーティンの期待反応", target: "RRB", source: "child", feature: "ges_child_pointing_candidate_per_min", n: null, rho: 0.379769, q: 0.035964 },
  { task: "共同注意への反応", target: "SA", source: "examiner", feature: "examiner_turn_rate_per_min", n: 40, rho: -0.355083, q: 0.02997 },
  { task: "共同注意への反応", target: "SA", source: "dyad", feature: "dyad_chain_ge4_per_min", n: 40, rho: -0.349493, q: 0.033966 },
  { task: "共同注意への反応", target: "SA", source: "dyad", feature: "dyad_silence_frac", n: 40, rho: 0.335839, q: 0.033966 },
  { task: "実演課題", target: "CSS", source: "examiner", feature: "examiner_turn_words_mean", n: 26, rho: -0.464977, q: 0.021978 },
  { task: "実演課題", target: "RRB", source: "child", feature: "ges_child_leaning_away_per_min", n: 29, rho: 0.477184, q: 0.008991 },
  { task: "実演課題", target: "SA", source: "examiner", feature: "ges_examiner_fidgeting_per_min", n: 29, rho: 0.409062, q: 0.048951 },
  { task: "実演課題", target: "SA", source: "child", feature: "win_child_turn_rate_delta", n: 29, rho: 0.385204, q: 0.048951 },
  { task: "実演課題", target: "SA", source: "examiner", feature: "examiner_turn_words_mean", n: 26, rho: -0.42196, q: 0.048951 },
  { task: "絵の叙述", target: "RRB", source: "child", feature: "ges_child_leaning_away_per_min", n: 21, rho: 0.527976, q: 0.021978 },
  { task: "絵の叙述", target: "RRB", source: "child", feature: "win_child_yawvel_delta", n: 21, rho: 0.507811, q: 0.021978 },
  { task: "絵の叙述", target: "SA", source: "dyad", feature: "dyad_motion_share_child", n: 21, rho: 0.568266, q: 0.026973 },
  { task: "絵の叙述", target: "SA", source: "examiner", feature: "examiner_sway_y", n: 21, rho: -0.517318, q: 0.043457 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "ges_child_hand_near_face_per_min", n: 19, rho: 0.581485, q: 0.02972 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "child_turn_words_mean", n: 17, rho: 0.646782, q: 0.02972 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "child_speech_frac", n: 18, rho: 0.60021, q: 0.02972 },
  { task: "本のストーリーの説明", target: "CSS", source: "dyad", feature: "rsp_after_exam_gap_med", n: 18, rho: -0.534922, q: 0.02972 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "child_backchannel_frac", n: 17, rho: -0.556385, q: 0.033566 },
  { task: "本のストーリーの説明", target: "CSS", source: "examiner", feature: "win_examiner_speech_frac_delta", n: 19, rho: -0.487568, q: 0.041958 },
  { task: "本のストーリーの説明", target: "CSS", source: "dyad", feature: "dyad_turn_dur_ratio", n: 16, rho: 0.525574, q: 0.041958 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "child_backchannel_frac", n: 17, rho: -0.793746, q: 0.008991 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "child_speech_frac", n: 18, rho: 0.703163, q: 0.011988 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "ges_child_hand_near_face_per_min", n: 19, rho: 0.715467, q: 0.011988 },
  { task: "本のストーリーの説明", target: "RRB", source: "examiner", feature: "win_examiner_speech_frac_delta", n: 19, rho: -0.604141, q: 0.012587 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "child_turn_words_mean", n: 17, rho: 0.631784, q: 0.012587 },
  { task: "本のストーリーの説明", target: "RRB", source: "dyad", feature: "rsp_after_exam_gap_med", n: 18, rho: -0.584367, q: 0.023976 },
  { task: "本のストーリーの説明", target: "RRB", source: "dyad", feature: "dyad_turn_dur_ratio", n: 16, rho: 0.577618, q: 0.028971 },
  { task: "本のストーリーの説明", target: "RRB", source: "examiner", feature: "examiner_backchannel_frac", n: 18, rho: 0.532172, q: 0.028971 },
  { task: "本のストーリーの説明", target: "RRB", source: "dyad", feature: "dyad_child_speech_share", n: 18, rho: 0.529656, q: 0.028971 },
  { task: "本のストーリーの説明", target: "SA", source: "examiner", feature: "ges_examiner_pointing_candidate_per_min", n: 19, rho: -0.595256, q: 0.01049 },
  { task: "本のストーリーの説明", target: "SA", source: "examiner", feature: "ges_examiner_arm_extended_per_min", n: 19, rho: -0.631737, q: 0.01049 },
  { task: "本のストーリーの説明", target: "SA", source: "child", feature: "child_turn_words_mean", n: 17, rho: 0.539764, q: 0.02997 },
];

function fmtRho(x: number): string {
  return (x >= 0 ? "+" : "") + x.toFixed(3);
}

function fmtQ(x: number): string {
  return x.toFixed(4);
}

export default function Exp1V6TwosidedCanvas() {
  const hitRows = HITS.map((h) => [
    h.task,
    h.target,
    h.source,
    <Code key={h.feature}>{h.feature}</Code>,
    h.n == null ? "—" : String(h.n),
    fmtRho(h.rho),
    fmtQ(h.q),
  ]);

  const rowTone = HITS.map((h) => (h.rho < 0 ? "danger" : "info")) as Array<
    "danger" | "info"
  >;

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>実験1 · v6再分割 · 両側 Spearman</H1>
        <Text tone="secondary">
          2026-08-20 · 課題区間は v6（A–B–A 補正, untag 281.8 分）· 安定度は同符号
          |ρ|≥0.20 · 置換は両側 · BH は課題×得点
        </Text>
        <Row gap={8} wrap>
          <Pill tone="info">stage A 2211</Pill>
          <Pill tone="info">candidates 120</Pill>
          <Pill tone="success">FDR hits 85</Pill>
          <Pill tone="warning">ρ+ 36 / ρ− 49</Pill>
        </Row>
      </Stack>

      <Callout tone="warning" title="前回 HTML では開けなかった理由">
        先に置いた <Code>docs/exp1-hits.html</Code>{" "}
        はブラウザ用の静的コピーで、Cursor 本体の Canvas
        ではありません。この画面がチャット横で開く正式な Canvas です。
      </Callout>

      <H2>前回（正側のみ）との比較</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader trailing={<Pill size="sm">2026-08-16/17</Pill>}>
            正側のみ · v6前
          </CardHeader>
          <CardBody>
            <Grid columns={3} gap={10}>
              <Stat value="2211" label="ステージA" />
              <Stat value="128" label="候補" />
              <Stat value="110" label="FDR 当たり" tone="info" />
            </Grid>
            <Text size="small" tone="secondary" style={{ marginTop: 10 }}>
              ρ&gt;0 のみ · 片側置換 · 生区間 · 人数多い5課題 38 · ASR依存 17
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill tone="success" size="sm">今回</Pill>}>
            両側 · v6
          </CardHeader>
          <CardBody>
            <Grid columns={3} gap={10}>
              <Stat value="2211" label="ステージA" />
              <Stat value="120" label="候補" tone="warning" />
              <Stat value="85" label="FDR 当たり" tone="warning" />
            </Grid>
            <Text size="small" tone="secondary" style={{ marginTop: 10 }}>
              同符号 |ρ|≥0.20 · 両側置換 · untag 281.8 分 · 散布図 85 枚
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns="1.2fr 1fr" gap={16}>
        <Stack gap={8}>
          <H2>符号の内訳（85 本）</H2>
          <BarChart
            categories={["正の相関", "負の相関"]}
            series={[{ name: "FDR hits", data: [36, 49], tone: "info" }]}
            height={180}
            horizontal
          />
          <Text size="small" tone="tertiary">
            Source: outputs/exp1/meta.json · 2026-08-20 two-sided + v6 run
          </Text>
        </Stack>
        <Stack gap={8}>
          <H2>当たりの構成比</H2>
          <PieChart
            data={[
              { label: "正 ρ (36)", value: 36 },
              { label: "負 ρ (49)", value: 49 },
            ]}
            height={180}
          />
          <Text size="small" tone="tertiary">
            過半が負。正側だけの前回とは一覧の顔ぶれが大きく違う。
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>何が効いて数が変わったか</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>両側化</CardHeader>
          <CardBody>
            <Text>
              片側 p は「ρ 以上」、両側 p は「|ρ| 以上」。対称ならだいたい{" "}
              <Code>p_two ≈ 2 · p_one</Code>。さらに負の候補が増えて BH
              の競争相手も増える。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>v6 再分割</CardHeader>
          <CardBody>
            <Text>
              A–B–A 誤タグを消し、3 分以上の外れ切れ端は untag（合計 281.8
              分）。課題に載る映像が前回と違うので、同じ列でも ρ が動きうる。
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <Row gap={8} align="center" justify="space-between">
          <H2>当たり一覧（ログから復元 46 / 85）</H2>
          <Pill tone="warning" size="sm">
            不完全
          </Pill>
        </Row>
        <Callout tone="info" title="表について">
          このクラウド環境に本番の <Code>hits_main.csv</Code>{" "}
          が無く、前セッションのログから先頭・末尾の 46
          行だけ復元しています。ごっこ・相互遊び・構成・呼名などの中盤は欠落。完全表は再実行環境の{" "}
          <Code>outputs/exp1/hits_main.csv</Code> を正とします。
        </Callout>
        <Table
          headers={["課題", "得点", "情報源", "特徴", "n", "ρ", "q"]}
          rows={hitRows}
          rowTone={rowTone}
          columnAlign={["left", "left", "left", "left", "right", "right", "right"]}
          stickyHeader
          striped
          style={{ maxHeight: 420 }}
        />
        <Text size="small" tone="tertiary">
          行の色点: 青寄り=正の ρ / 赤寄り=負の ρ · Source: terminal log recovery ·
          not the full 85-row CSV
        </Text>
      </Stack>

      <Stack gap={4}>
        <H3>手続きメモ</H3>
        <Text size="small" tone="secondary">
          半分割り安定度（同符号 |ρ|≥0.20）→ 課題×得点の null 95% 閾値 → 両側置換
          1000 回 → BH q&lt;0.05。特徴表は v6 再分割後。
        </Text>
      </Stack>
    </Stack>
  );
}
