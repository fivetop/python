Rhyme Score
> '소리 나는 대로 발음한' 두 토큰의 발음 유사도를 측정하기 위함이다
> 최고점은 8점, 최하점은 0점이다
> 각 토큰의 마지막 글자, 마지막에서 두번째 글자를 비교하여 점수를 매긴다
  - 만약 둘 중 하나의 토큰이라도 토큰의 길이가 1(한 글자짜리 토큰)이면 가장 마지막 글자끼리만을 비교한다
> 점수 규칙
  - (1) 모음 평가
    - 5점 획득: 모음이 완전히 같은 경우
    - 3점 획득: 두 토큰의 모음이 발음이 비슷해서 같은 '유사 모음 그룹'에 포함될 경우
      ***'유사 모음 그룹': ['ㅏ', 'ㅘ', 'ㅑ'], ['ㅓ', 'ㅕ', 'ㅝ'], ['ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅙ', 'ㅚ', 'ㅞ'], ['ㅗ', 'ㅛ'], ['ㅜ', 'ㅠ', 'ㅡ'], ['ㅟ', 'ㅢ', 'ㅣ']
    - 0점 획득(실격): 모음이 유사 그룹에도 속하지 않는 경우 0점으로 실격, 평가를 (2) 자음 평가를 하지 않고 평가를 종료한다
  - (2) 자음 평가
    - 3점 획득: 두 토큰이 '모두 종성(받침)을 가지고 있거나', '모두 종성을 가지고 있지 않거나'의 두 경우
      - 1점 차감: '모두 종성을 가지고 있거나'의 경우에 한해서 '한 종성은 울림 소리에 포함되지만 나머지 한 종성은 울림 소리에 포함되는 경우'
    - 1점 획득: 위 두 경우에 해당하지 않는 경우
  - 가중 평균
    - 운율의 특성상 마지막 글자의 통일성이 더 중요하다
    - 마지막 글자를 L, 그 앞 글자를 postL이라고 할 때, Ryhme Score는 다음과 같다.
        Rhyme Score = Score(postL)*0.25 + Score(L)*0.75



