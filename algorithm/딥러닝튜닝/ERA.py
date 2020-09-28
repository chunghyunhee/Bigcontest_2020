# feature engineering

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# 추가한 새로운 feature
def cal_sm_1(new_dt):
  PA = new_dt['PA'] #타석
  AB = new_dt['AB'] #타수
  RBI = new_dt['RBI'] #타점
  RUN = new_dt['RUN'] #득점
  HIT = new_dt['HIT'] #총안타
  H2 = new_dt['H2'] #2루타
  H3 = new_dt['H3'] #3루타
  HR = new_dt['HR'] #홈런
  SB = new_dt['SB'] #도루
  CS = new_dt['CS'] #도루실패
  SH = new_dt['SH'] #희생타
  SF = new_dt['SF'] #희생플라이
  IB = new_dt['IB'] #고의4구
  HP = new_dt['HP'] #빈볼
  BB = new_dt['BB'] #총사구
  KK = new_dt['KK'] #삼진
  GD = new_dt['GD'] #병살타
  ERR = new_dt['ERR'] #실책

  # 타율 계산
  new_dt['AVG'] = HIT/AB

  # 순수 1루타 계산
  new_dt['H1'] = HIT - H2 - H3 - HR
  H1 = new_dt['H1']

  # OBP(출루율)
  new_dt['OBP'] = (HIT+BB+HP)/(AB+BB+SF+HP)
  # SLG(장타율)
  new_dt['SLG'] = (H1+2*H2+3*H3+4*HR)/AB

  # OPS = OBP+SLG
  OBP = new_dt['OBP']
  SLG = new_dt['SLG']
  new_dt['OPS'] = OBP + SLG

  # GPA = (1.8*OBP+SLG)/4 OPS의 단점을 보완
  new_dt['GPA'] = (1.8*new_dt['OBP'] + new_dt['SLG'])/4

  # IsoP = SLG - AVG 순수장타율
  AVG = new_dt['AVG']
  new_dt['IsoP'] = SLG - AVG

  # RC(Run Created)
  new_dt['RC'] = ((HIT+BB-CS+HP-GD) * ((H1+2*H2+3*H3+4*HR)+0.26*(BB-IB+HP)) + (0.52*SH+SF+SB)) / (AB+BB+HP+SH+SF)

  # XR (eXtrapolated Runs, 추정득점(타자의 득점공헌도))
  new_dt['XR'] = (H1*0.5 + H2*0.72 + H3*1.04 + HR*1.44 + (HP+BB-IB)*0.34 + IB*0.25 + SB*0.18 - CS*0.32
                  - (AB-HIT-KK)*0.09 - KK*0.098 - GD*0.37 + SF*0.37 + SH*0.04 )

  # BABIP = (총 안타수-홈런)/(타수-삼진-홈런+희생플라이)
  new_dt['BABIP'] = (HIT-HR)/(AB-KK-HR+SF)

  # OBP = (HIT+BB+HP)/(AB+BB+SF+HP) 출루율
  new_dt['OBP'] = (HIT+BB+HP)/(AB+BB+SF+HP)

  # SLG = (HIT+2*H2+3*H3+4*HR)/AB 장타율
  new_dt['SLG'] = (H1+2*H2+3*H3+4*HR)/AB

  # OPS+ = 100*((OBP/lgOBP+SLG/lgSLG)-1)[/BPF]
  # lgOBP: 리그 평균 출루율 / lgSLG: 리그 평균 장타율
  lgOBP = new_dt['OBP'].mean() # 결측값(NaN) 제외하고 리그 평균 계산
  lgSLG = new_dt['SLG'].mean()
  new_dt['OPS_plus'] = 100*((new_dt['OBP']/lgOBP+new_dt['SLG']/lgSLG)-1)

  # 타수가 극단적으로 적은 타자들로 인해 기록이 부정확할 확률이 높음
  # OPS+ 계산에는 원래 파크팩터를 사용하기도 함. 그러나 시즌 총 기록에서 경기별 패크팩터를 어떻게 대입할지는 모르겠음.

  return new_dt



# 2016 데이터로 확인
df = pd.read_csv('2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2016.csv')
df.tail()



# 쓰이지 않는 피처 제거
df.drop(['G_ID'], axis=1, inplace=True)
df.drop(['GDAY_DS'], axis=1, inplace=True)
df.drop(['HEADER_NO'], axis=1, inplace=True)
df.drop(['TB_SC'], axis=1, inplace=True)



import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

team_label = encoder.fit_transform(df['T_ID'])
vs_team_label = encoder.fit_transform(df['VS_T_ID'])

print(team_label)
print(vs_team_label)



df['T_ID'] = team_label
df['VS_T_ID'] = vs_team_label

df.tail()


df.info() # null, object check


#  팀 Set 제작
team = set()
for i in df['T_ID']:
  team.add(i)
print("2016년 참여 팀 수 :", len(team))


df = cal_sm_1(df)
df.head()

cl = df.columns.tolist()
arr = []
df18 = pd.DataFrame(index=range(0, 0), columns=cl)
print(df18)
for i in team:
  # 선수 별로 새로운 Dataframe 생성
  hitter_df = df[df['T_ID'] == i]
  hitter_df.reset_index(drop=True, inplace=True)

  # 경기 데이터가 18개 이상이라면
  count = len(hitter_df) // 18

  if count != 0:
    # print(count)
    for k in range(count):
      s = k * 18
      # print(s,'부터',s+17,'까지의 합')

      total = hitter_df.loc[s] + hitter_df.loc[s + 1] + hitter_df.loc[s + 2] + hitter_df.loc[s + 3] + hitter_df.loc[
        s + 4] + hitter_df.loc[s + 5] + hitter_df.loc[s + 6] + hitter_df.loc[s + 7] + hitter_df.loc[s + 8] + \
              hitter_df.loc[s + 9] + hitter_df.loc[s + 10] + hitter_df.loc[s + 11] + hitter_df.loc[s + 12] + \
              hitter_df.loc[s + 13] + hitter_df.loc[s + 14] + hitter_df.loc[s + 15] + hitter_df.loc[s + 16] + \
              hitter_df.loc[s + 17]
      total = total / 18

      df18 = df18.append(pd.Series(total, index=df18.columns), ignore_index=True)
      # print(k, end=' ')
      if k == 0:
        pass
      else:
        arr.append(total['AVG'])
    arr.append(0)


df18['label']=arr
df18 = df18[df18['label']!=0]
df18_2 = df18.copy()

df18_2 = df18_2.fillna(0)
df18_2.reset_index(drop=True, inplace=True)


df18_2


df18_3 = df18_2[['AB', 'HIT', 'KK', 'AVG', 'OPS', 'RC', 'BABIP', 'OPS_plus', 'label']]
df18_3



from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

y_target = df18_3['label']
X_data = df18_3.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.3, random_state=123)


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, y_pred)))

