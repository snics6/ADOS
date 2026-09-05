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
  { task: "呼名反応", target: "RRB", source: "dyad", feature: "dyad_chain_ge4_per_min", n: 22, rho: 0.5602976772145419, q: 0.0079920079920079 },
  { task: "ごっこ遊び", target: "CSS", source: "examiner", feature: "win_examiner_speech_frac_delta", n: 54, rho: -0.4222948798740339, q: 0.0119880119880119 },
  { task: "ごっこ遊び", target: "CSS", source: "examiner", feature: "win_examiner_turn_rate_delta", n: 54, rho: -0.4162649656497245, q: 0.0119880119880119 },
  { task: "ごっこ遊び", target: "CSS", source: "examiner", feature: "ges_examiner_pointing_candidate_per_min", n: 54, rho: -0.3039994717824842, q: 0.0374625374625374 },
  { task: "ごっこ遊び", target: "CSS", source: "child", feature: "win_child_turn_rate_delta", n: 54, rho: -0.3176483468025632, q: 0.0374625374625374 },
  { task: "ごっこ遊び", target: "CSS", source: "examiner", feature: "ges_examiner_arm_extended_per_min", n: 54, rho: -0.27372505160425, q: 0.0409590409590409 },
  { task: "ごっこ遊び", target: "CSS", source: "child", feature: "win_child_speech_frac_delta", n: 54, rho: -0.2869180523546067, q: 0.0409590409590409 },
  { task: "ごっこ遊び", target: "RRB", source: "examiner", feature: "win_examiner_turn_rate_delta", n: 54, rho: -0.4707453530643739, q: 0.0089910089910089 },
  { task: "ごっこ遊び", target: "RRB", source: "examiner", feature: "ges_examiner_leaning_away_per_min", n: 54, rho: -0.4254241240594746, q: 0.0134865134865134 },
  { task: "ごっこ遊び", target: "RRB", source: "dyad", feature: "rsp_long_silence_frac", n: 54, rho: -0.4254928974368241, q: 0.0149850149850149 },
  { task: "ごっこ遊び", target: "RRB", source: "child", feature: "win_child_speech_frac_delta", n: 54, rho: -0.3398898518156431, q: 0.0179820179820179 },
  { task: "ごっこ遊び", target: "RRB", source: "child", feature: "win_child_turn_rate_delta", n: 54, rho: -0.3493613393332985, q: 0.0179820179820179 },
  { task: "ごっこ遊び", target: "RRB", source: "examiner", feature: "win_examiner_speech_frac_delta", n: 54, rho: -0.3333790139305616, q: 0.0179820179820179 },
  { task: "ごっこ遊び", target: "RRB", source: "examiner", feature: "examiner_turn_rate_per_min", n: 54, rho: 0.2721289093820171, q: 0.0439560439560439 },
  { task: "ごっこ遊び", target: "RRB", source: "examiner", feature: "ges_examiner_pointing_candidate_per_min", n: 54, rho: -0.2781627643975564, q: 0.0439560439560439 },
  { task: "ごっこ遊び", target: "RRB", source: "dyad", feature: "rsp_after_exam_gap_p90", n: 54, rho: -0.2869591512313695, q: 0.0439560439560439 },
  { task: "ごっこ遊び", target: "SA", source: "examiner", feature: "win_examiner_speech_frac_delta", n: 54, rho: -0.4140267788761179, q: 0.0059940059940059 },
  { task: "ごっこ遊び", target: "SA", source: "examiner", feature: "win_examiner_turn_rate_delta", n: 54, rho: -0.3184592421535803, q: 0.0179820179820179 },
  { task: "共同で行う相互的な遊び", target: "CSS", source: "dyad", feature: "dyad_chain_max", n: 47, rho: -0.3612432990188096, q: 0.0179820179820179 },
  { task: "共同で行う相互的な遊び", target: "CSS", source: "dyad", feature: "rsp_child_reply_frac", n: 51, rho: -0.3589961620808822, q: 0.0179820179820179 },
  { task: "共同で行う相互的な遊び", target: "CSS", source: "child", feature: "win_child_yawvel_delta", n: 50, rho: 0.2989250417038659, q: 0.0379620379620379 },
  { task: "共同で行う相互的な遊び", target: "RRB", source: "examiner", feature: "ges_examiner_pointing_candidate_per_min", n: 51, rho: 0.4753683399834631, q: 0.0059940059940059 },
  { task: "共同で行う相互的な遊び", target: "RRB", source: "examiner", feature: "ges_examiner_arm_extended_per_min", n: 51, rho: 0.384978722766432, q: 0.0089910089910089 },
  { task: "共同で行う相互的な遊び", target: "RRB", source: "dyad", feature: "rsp_child_reply_frac", n: 51, rho: -0.3884964596033879, q: 0.0119880119880119 },
  { task: "共同で行う相互的な遊び", target: "RRB", source: "child", feature: "txt_child_utt_char_cv", n: 49, rho: 0.3441058281178129, q: 0.0227772227772227 },
  { task: "共同で行う相互的な遊び", target: "RRB", source: "child", feature: "win_child_turn_rate_delta", n: 51, rho: 0.333107414687828, q: 0.0227772227772227 },
  { task: "共同で行う相互的な遊び", target: "SA", source: "dyad", feature: "dyad_chain_max", n: 47, rho: -0.3313829435255058, q: 0.0494505494505494 },
  { task: "共同で行う相互的な遊び", target: "SA", source: "child", feature: "win_child_yawvel_delta", n: 50, rho: 0.3010478802624418, q: 0.0494505494505494 },
  { task: "共同で行う相互的な遊び", target: "SA", source: "dyad", feature: "dyad_turn_rate_per_min", n: 47, rho: -0.2919605387421286, q: 0.0499500499500499 },
  { task: "共同注意への反応", target: "CSS", source: "dyad", feature: "dyad_turn_rate_per_min", n: 40, rho: -0.4500934373812044, q: 0.0299700299700299 },
  { task: "共同注意への反応", target: "RRB", source: "examiner", feature: "ges_examiner_hand_near_face_per_min", n: 42, rho: 0.3645430223944339, q: 0.0259740259740259 },
  { task: "共同注意への反応", target: "RRB", source: "dyad", feature: "rsp_after_exam_gap_med", n: 41, rho: -0.3898183911255919, q: 0.0259740259740259 },
  { task: "共同注意への反応", target: "RRB", source: "dyad", feature: "rsp_long_silence_frac", n: 41, rho: -0.3470607959864821, q: 0.0319680319680319 },
  { task: "共同注意への反応", target: "SA", source: "dyad", feature: "dyad_turn_rate_per_min", n: 40, rho: -0.479513237316312, q: 0.0119880119880119 },
  { task: "共同注意への反応", target: "SA", source: "child", feature: "ges_child_hand_near_face_per_min", n: 42, rho: 0.4365178641873353, q: 0.0149850149850149 },
  { task: "共同注意への反応", target: "SA", source: "child", feature: "child_turn_rate_per_min", n: 40, rho: -0.3916858078570387, q: 0.0299700299700299 },
  { task: "共同注意への反応", target: "SA", source: "examiner", feature: "examiner_turn_rate_per_min", n: 40, rho: -0.3550831841940977, q: 0.0299700299700299 },
  { task: "共同注意への反応", target: "SA", source: "dyad", feature: "dyad_chain_ge4_per_min", n: 40, rho: -0.3494931568075489, q: 0.0339660339660339 },
  { task: "共同注意への反応", target: "SA", source: "dyad", feature: "dyad_silence_frac", n: 40, rho: 0.3358385057733761, q: 0.0339660339660339 },
  { task: "実演課題", target: "CSS", source: "examiner", feature: "examiner_turn_words_mean", n: 26, rho: -0.4649771090063622, q: 0.0219780219780219 },
  { task: "実演課題", target: "RRB", source: "child", feature: "ges_child_leaning_away_per_min", n: 29, rho: 0.4771844575066214, q: 0.0089910089910089 },
  { task: "実演課題", target: "SA", source: "examiner", feature: "examiner_turn_words_mean", n: 26, rho: -0.4219600198062397, q: 0.0489510489510489 },
  { task: "実演課題", target: "SA", source: "examiner", feature: "ges_examiner_fidgeting_per_min", n: 29, rho: 0.4090620959594241, q: 0.0489510489510489 },
  { task: "実演課題", target: "SA", source: "child", feature: "win_child_turn_rate_delta", n: 29, rho: 0.3852042823433216, q: 0.0489510489510489 },
  { task: "絵の叙述", target: "RRB", source: "child", feature: "ges_child_leaning_away_per_min", n: 21, rho: 0.5279758608505342, q: 0.0219780219780219 },
  { task: "絵の叙述", target: "RRB", source: "child", feature: "win_child_yawvel_delta", n: 21, rho: 0.5078112915959115, q: 0.0219780219780219 },
  { task: "絵の叙述", target: "SA", source: "dyad", feature: "dyad_motion_share_child", n: 21, rho: 0.5682658606376936, q: 0.0269730269730269 },
  { task: "絵の叙述", target: "SA", source: "examiner", feature: "examiner_sway_y", n: 21, rho: -0.5173178869253486, q: 0.0434565434565434 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "child_speech_frac", n: 18, rho: 0.6002104909091768, q: 0.0297202797202797 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "child_turn_words_mean", n: 17, rho: 0.6467817173476824, q: 0.0297202797202797 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "ges_child_hand_near_face_per_min", n: 19, rho: 0.5814854697356918, q: 0.0297202797202797 },
  { task: "本のストーリーの説明", target: "CSS", source: "dyad", feature: "rsp_after_exam_gap_med", n: 18, rho: -0.534921845712868, q: 0.0297202797202797 },
  { task: "本のストーリーの説明", target: "CSS", source: "child", feature: "child_backchannel_frac", n: 17, rho: -0.556385060001845, q: 0.0335664335664335 },
  { task: "本のストーリーの説明", target: "CSS", source: "dyad", feature: "dyad_turn_dur_ratio", n: 16, rho: 0.5255738049628411, q: 0.0419580419580419 },
  { task: "本のストーリーの説明", target: "CSS", source: "examiner", feature: "win_examiner_speech_frac_delta", n: 19, rho: -0.4875680046783928, q: 0.0419580419580419 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "child_backchannel_frac", n: 17, rho: -0.7937460244969139, q: 0.0089910089910089 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "child_speech_frac", n: 18, rho: 0.7031633602207793, q: 0.0119880119880119 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "ges_child_hand_near_face_per_min", n: 19, rho: 0.7154669753117213, q: 0.0119880119880119 },
  { task: "本のストーリーの説明", target: "RRB", source: "child", feature: "child_turn_words_mean", n: 17, rho: 0.631784356766515, q: 0.0125874125874125 },
  { task: "本のストーリーの説明", target: "RRB", source: "examiner", feature: "win_examiner_speech_frac_delta", n: 19, rho: -0.6041407644655868, q: 0.0125874125874125 },
  { task: "本のストーリーの説明", target: "RRB", source: "dyad", feature: "rsp_after_exam_gap_med", n: 18, rho: -0.5843665341002411, q: 0.0239760239760239 },
  { task: "本のストーリーの説明", target: "RRB", source: "dyad", feature: "dyad_child_speech_share", n: 18, rho: 0.5296555180883792, q: 0.0289710289710289 },
  { task: "本のストーリーの説明", target: "RRB", source: "dyad", feature: "dyad_turn_dur_ratio", n: 16, rho: 0.5776177144209592, q: 0.0289710289710289 },
  { task: "本のストーリーの説明", target: "RRB", source: "examiner", feature: "examiner_backchannel_frac", n: 18, rho: 0.5321717850012481, q: 0.0289710289710289 },
  { task: "本のストーリーの説明", target: "SA", source: "examiner", feature: "ges_examiner_arm_extended_per_min", n: 19, rho: -0.6317366060320145, q: 0.0104895104895104 },
  { task: "本のストーリーの説明", target: "SA", source: "examiner", feature: "ges_examiner_pointing_candidate_per_min", n: 19, rho: -0.5952560414583348, q: 0.0104895104895104 },
  { task: "本のストーリーの説明", target: "SA", source: "child", feature: "child_turn_words_mean", n: 17, rho: 0.5397639049260996, q: 0.0299700299700299 },
  { task: "誕生日パーティ", target: "CSS", source: "examiner", feature: "win_examiner_turn_rate_delta", n: 58, rho: 0.2958844560488144, q: 0.0219780219780219 },
  { task: "誕生日パーティ", target: "SA", source: "examiner", feature: "win_examiner_turn_rate_delta", n: 58, rho: 0.3302538978477711, q: 0.0269730269730269 },
  { task: "誕生日パーティ", target: "SA", source: "examiner", feature: "examiner_backchannel_frac", n: 56, rho: -0.2921585423236513, q: 0.0359640359640359 },
  { task: "誕生日パーティ", target: "SA", source: "examiner", feature: "examiner_pitch_reversal_rate", n: 56, rho: -0.2883541301111929, q: 0.0359640359640359 },
  { task: "おやつ", target: "RRB", source: "child", feature: "child_sway_y", n: 56, rho: -0.279334848923944, q: 0.0467532467532467 },
  { task: "おやつ", target: "RRB", source: "examiner", feature: "examiner_roll_range", n: 56, rho: -0.3165338819184566, q: 0.0467532467532467 },
  { task: "おやつ", target: "RRB", source: "examiner", feature: "examiner_sway_y", n: 56, rho: -0.2790107527991129, q: 0.0467532467532467 },
  { task: "おやつ", target: "RRB", source: "dyad", feature: "rsp_after_exam_gap_med", n: 56, rho: 0.2763164400397037, q: 0.0467532467532467 },
  { task: "おやつ", target: "RRB", source: "child", feature: "txt_child_utt_char_cv", n: 56, rho: -0.272024680774974, q: 0.0467532467532467 },
  { task: "おやつ", target: "SA", source: "child", feature: "txt_child_question_frac", n: 56, rho: 0.391675065940727, q: 0.0199800199800199 },
  { task: "おやつ", target: "SA", source: "dyad", feature: "rsp_child_reply_frac", n: 56, rho: -0.3187212695109455, q: 0.0339660339660339 },
  { task: "おやつ", target: "SA", source: "dyad", feature: "rsp_after_exam_gap_med", n: 56, rho: 0.2857286221728373, q: 0.0466200466200466 },
  { task: "ものを用いたルーティンの期待反応", target: "CSS", source: "child", feature: "child_backchannel_frac", n: 45, rho: -0.3326330739932116, q: 0.0319680319680319 },
  { task: "ものを用いたルーティンの期待反応", target: "CSS", source: "dyad", feature: "rsp_after_exam_gap_med", n: 45, rho: -0.3365008578142531, q: 0.0319680319680319 },
  { task: "ものを用いたルーティンの期待反応", target: "RRB", source: "child", feature: "ges_child_pointing_candidate_per_min", n: 45, rho: 0.3797687169331182, q: 0.0359640359640359 },
  { task: "ものを用いたルーティンの期待反応", target: "RRB", source: "examiner", feature: "ges_examiner_pose_frame_frac", n: 45, rho: 0.3023467493565692, q: 0.0494505494505494 },
  { task: "ものを用いたルーティンの期待反応", target: "SA", source: "child", feature: "ges_child_pointing_candidate_per_min", n: 45, rho: -0.3196662841647039, q: 0.0329670329670329 },
  { task: "ものを用いたルーティンの期待反応", target: "SA", source: "child", feature: "txt_child_question_frac", n: 45, rho: 0.3225863662346805, q: 0.0329670329670329 },
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
    <Code>{h.feature}</Code>,
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
          <Pill active>stage A 2211</Pill>
          <Pill active>candidates 120</Pill>
          <Pill active>FDR hits 85</Pill>
          <Pill active>ρ+ 36 / ρ− 49</Pill>
        </Row>
      </Stack>

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
          <CardHeader trailing={<Pill active size="sm">今回</Pill>}>
            両側 · v6
          </CardHeader>
          <CardBody>
            <Grid columns={3} gap={10}>
              <Stat value="2211" label="ステージA" />
              <Stat value="120" label="候補" tone="warning" />
              <Stat value="85" label="FDR 当たり" tone="warning" />
            </Grid>
            <Text size="small" tone="secondary" style={{ marginTop: 10 }}>
              同符号 |ρ|≥0.20 · 両側置換 · untag 281.8 分 · 人数多い5課題 46 · ASR依存 4 ·
              散布図 85 枚
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
            Source: outputs/exp1/hits_main.csv · 2026-08-20 two-sided + v6 run
          </Text>
        </Stack>
        <Stack gap={8}>
          <H2>当たりの構成比</H2>
          <PieChart
            data={[
              { label: "正 ρ (36)", value: 36 },
              { label: "負 ρ (49)", value: 49 },
            ]}
            size={180}
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
          <H2>当たり一覧（全 85 本）</H2>
          <Pill active size="sm">
            hits_main.csv
          </Pill>
        </Row>
        <Callout tone="info" title="出典">
          ローカルの <Code>outputs/exp1/hits_main.csv</Code> 
          をそのまま表示しています。行の色点は正の ρ / 負の ρ を示します。
        </Callout>
        <Table
          headers={["課題", "得点", "情報源", "特徴", "n", "ρ", "q"]}
          rows={hitRows}
          rowTone={rowTone}
          columnAlign={["left", "left", "left", "left", "right", "right", "right"]}
          stickyHeader
          striped
          style={{ maxHeight: 480 }}
        />
        <Text size="small" tone="tertiary">
          Source: outputs/exp1/hits_main.csv · q = BH within task×target · FDR hits = 85
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
