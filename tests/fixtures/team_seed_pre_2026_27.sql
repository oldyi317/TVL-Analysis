-- 球隊身分種子（版本化；DB 的 teams/team_aliases 為執行期單一來源）。可重複執行（UPSERT）。
-- 新隊伍／改名：改本檔後以 apply_team_seed 套用（見 docs/ops/season-switch.md）。

INSERT INTO teams (team_id, team_name, gender, short_name, official_name, color, is_active) VALUES
    (1, '屏東台電', 'M', '屏東台電', '台灣電力公司男子排球隊', '#1B9E77', 1),
    (2, '雲林美津濃', 'M', '雲林美津濃', '美津濃男子排球隊', '#D95F02', 1),
    (4, '臺北國北獅', 'M', '臺北國北獅', '臺北國北獅', '#7570B3', 1),
    (5, '桃園臺產', 'M', '桃園臺產', '桃園臺產隼鷹排球隊', '#E7298A', 1),
    (7, '獅子王', 'M', '獅子王', '獅子王', '#66A61E', 1),
    (3, '臺北鯨華', 'F', '臺北鯨華', '臺北鯨華女子排球隊', '#E6AB02', 1),
    (4, '高雄台電', 'F', '高雄台電', '台灣電力公司女子排球隊', '#A6761D', 1),
    (5, '新北中纖', 'F', '新北中纖', '新北中國人纖企業女子排球隊', '#377EB8', 1),
    (7, '義力營造', 'F', '義力營造', '義力營造女子排球隊', '#984EA3', 1),
    (8, '彰化三大有線', 'M', '彰化三大有線', NULL, '#4DAF4A', 0),
    (9, '臺北Conti', 'M', '臺北Conti', NULL, '#666666', 0),
    (10, '連莊', 'M', '連莊', NULL, '#999999', 0)
ON CONFLICT(team_id, gender) DO UPDATE SET
    short_name=excluded.short_name, official_name=excluded.official_name,
    color=excluded.color, is_active=excluded.is_active;

INSERT INTO team_aliases (alias, team_id, gender, source) VALUES
    ('屏東台電', 1, 'M', 'short'),
    ('雲林美津濃', 2, 'M', 'short'),
    ('臺北國北獅', 4, 'M', 'short'),
    ('桃園臺產', 5, 'M', 'short'),
    ('獅子王', 7, 'M', 'short'),
    ('臺北鯨華', 3, 'F', 'short'),
    ('高雄台電', 4, 'F', 'short'),
    ('新北中纖', 5, 'F', 'short'),
    ('義力營造', 7, 'F', 'short'),
    ('臺北鯨華女子排球隊', 3, 'F', 'official'),
    ('新北中國人纖企業女子排球隊', 5, 'F', 'official'),
    ('台灣電力公司女子排球隊', 4, 'F', 'official'),
    ('義力營造女子排球隊', 7, 'F', 'official'),
    ('台灣電力公司男子排球隊', 1, 'M', 'official'),
    ('美津濃男子排球隊', 2, 'M', 'official'),
    ('桃園臺產隼鷹排球隊', 5, 'M', 'official'),
    ('桃園臺灣產險', 5, 'M', 'match'),
    ('臺中獅子王', 7, 'M', 'match'),
    ('屏東台電男排', 1, 'M', 'match'),
    ('高雄台電女排', 4, 'F', 'match'),
    ('彰化三大有線', 8, 'M', 'legacy'),
    ('臺北Conti', 9, 'M', 'legacy'),
    ('連莊', 10, 'M', 'legacy'),
    ('1', 1, 'M', 'ext_stats'),
    ('2', 2, 'M', 'ext_stats'),
    ('3', 7, 'M', 'ext_stats'),
    ('4', 4, 'M', 'ext_stats'),
    ('5', 5, 'M', 'ext_stats'),
    ('6', 4, 'F', 'ext_stats'),
    ('7', 3, 'F', 'ext_stats'),
    ('8', 5, 'F', 'ext_stats'),
    ('9', 7, 'F', 'ext_stats')
ON CONFLICT(alias) DO UPDATE SET
    team_id=excluded.team_id, gender=excluded.gender, source=excluded.source;
