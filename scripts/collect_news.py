"""
뉴스 데이터 수집 스크립트

사용법:
    python scripts/collect_news.py --query "부동산 정책" --num 100
"""

import argparse
import pandas as pd
from datetime import datetime
import json
import os


class NewsCollector:
    """뉴스 기사 수집 클래스"""

    def __init__(self):
        self.articles = []

    def collect_sample_articles(self, topic, num_articles=10):
        """
        샘플 기사 생성 (실제 API 연동 전 테스트용)

        Args:
            topic: 토픽 (예: "부동산 정책")
            num_articles: 생성할 기사 수
        """
        templates = {
            'support': [
                f"{topic}이(가) 경제 성장에 큰 도움이 될 것으로 기대된다. 전문가들은 긍정적으로 평가하고 있다.",
                f"이번 {topic}은(는) 국민의 삶의 질 향상에 기여할 것으로 보인다. 여러 지표가 개선되고 있다.",
                f"{topic}의 효과가 가시화되고 있다. 관련 업계도 만족스러운 반응을 보이고 있다.",
            ],
            'neutral': [
                f"정부가 {topic}을(를) 발표했다. 주요 내용은 다음과 같다.",
                f"{topic}에 대한 전문가들의 의견이 엇갈리고 있다. 향후 추이를 지켜봐야 할 것으로 보인다.",
                f"{topic} 관련 법안이 국회에 제출되었다. 현재 상임위원회에서 검토 중이다.",
            ],
            'oppose': [
                f"{topic}은(는) 현실을 제대로 반영하지 못했다. 전문가들은 부작용을 우려하고 있다.",
                f"이번 {topic}은(는) 실효성이 의심된다. 신중한 재검토가 필요하다는 지적이 나온다.",
                f"{topic}으로 인한 부담이 가중될 것이라는 우려가 제기되고 있다. 업계는 강력히 반발하고 있다.",
            ]
        }

        sources = ['조선일보', '한겨레', '연합뉴스', '중앙일보', '경향신문']

        for i in range(num_articles):
            stance_type = ['support', 'neutral', 'oppose'][i % 3]
            template = templates[stance_type][i % len(templates[stance_type])]

            article = {
                'text': template,
                'label': ['support', 'neutral', 'oppose'].index(stance_type),
                'source': sources[i % len(sources)],
                'date': datetime.now().strftime('%Y-%m-%d'),
                'topic': topic,
                'url': f'https://example.com/news/{i}',
                'note': 'sample'
            }
            self.articles.append(article)

        print(f"✅ {num_articles}개의 샘플 기사 생성 완료")

    def collect_from_naver_api(self, query, num_articles=100, client_id=None, client_secret=None):
        """
        Naver News API로 기사 수집

        Args:
            query: 검색 쿼리
            num_articles: 수집할 기사 수
            client_id: Naver API Client ID
            client_secret: Naver API Client Secret
        """
        if not client_id or not client_secret:
            print("⚠️  Naver API 키가 필요합니다.")
            print("https://developers.naver.com/apps/ 에서 발급받으세요.")
            return

        try:
            import requests

            url = "https://openapi.naver.com/v1/search/news.json"
            headers = {
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret
            }

            collected = 0
            start = 1

            while collected < num_articles:
                params = {
                    "query": query,
                    "display": min(100, num_articles - collected),
                    "start": start,
                    "sort": "date"
                }

                response = requests.get(url, headers=headers, params=params)
                if response.status_code != 200:
                    print(f"❌ API 오류: {response.status_code}")
                    break

                data = response.json()
                items = data.get('items', [])

                if not items:
                    break

                for item in items:
                    article = {
                        'text': self._clean_html(item.get('description', '')),
                        'label': None,  # 나중에 라벨링
                        'source': item.get('originallink', '').split('/')[2] if '//' in item.get('originallink', '') else '',
                        'date': item.get('pubDate', ''),
                        'topic': query,
                        'url': item.get('originallink', ''),
                        'note': 'unlabeled'
                    }
                    self.articles.append(article)
                    collected += 1

                start += len(items)

            print(f"✅ {collected}개 기사 수집 완료")

        except ImportError:
            print("❌ requests 패키지가 필요합니다: pip install requests")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    def _clean_html(self, text):
        """HTML 태그 제거"""
        import re
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        return text.strip()

    def save_to_csv(self, filename='data/collected_news.csv'):
        """CSV 파일로 저장"""
        if not self.articles:
            print("⚠️  저장할 기사가 없습니다.")
            return

        df = pd.DataFrame(self.articles)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ {len(self.articles)}개 기사를 {filename}에 저장했습니다.")

        # 통계 출력
        print(f"\n📊 수집 통계:")
        print(f"   전체: {len(df)}개")
        if 'label' in df.columns and df['label'].notna().any():
            print(f"   라벨별 분포:")
            label_names = {0: '옹호', 1: '중립', 2: '비판'}
            for label, count in df['label'].value_counts().items():
                label_name = label_names.get(label, '미분류')
                print(f"      {label_name}: {count}개")

    def save_to_json(self, filename='data/collected_news.json'):
        """JSON 파일로 저장"""
        if not self.articles:
            print("⚠️  저장할 기사가 없습니다.")
            return

        # 디렉토리 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)

        print(f"✅ {len(self.articles)}개 기사를 {filename}에 저장했습니다.")


def main():
    parser = argparse.ArgumentParser(description='뉴스 데이터 수집')
    parser.add_argument('--query', type=str, default='부동산 정책', help='검색 쿼리')
    parser.add_argument('--num', type=int, default=30, help='수집할 기사 수')
    parser.add_argument('--output', type=str, default='data/collected_news.csv', help='출력 파일명')
    parser.add_argument('--mode', type=str, choices=['sample', 'naver'], default='sample',
                        help='수집 모드: sample(샘플 생성) 또는 naver(Naver API)')
    parser.add_argument('--naver-id', type=str, help='Naver API Client ID')
    parser.add_argument('--naver-secret', type=str, help='Naver API Client Secret')

    args = parser.parse_args()

    collector = NewsCollector()

    if args.mode == 'sample':
        print(f"📝 샘플 데이터 생성 모드")
        print(f"   토픽: {args.query}")
        print(f"   개수: {args.num}개\n")
        collector.collect_sample_articles(args.query, args.num)
    elif args.mode == 'naver':
        print(f"🔍 Naver API 수집 모드")
        print(f"   검색어: {args.query}")
        print(f"   목표: {args.num}개\n")
        collector.collect_from_naver_api(
            args.query,
            args.num,
            args.naver_id,
            args.naver_secret
        )

    # 저장
    if args.output.endswith('.json'):
        collector.save_to_json(args.output)
    else:
        collector.save_to_csv(args.output)

    print(f"\n✅ 완료!")
    print(f"\n다음 단계:")
    print(f"1. {args.output} 파일을 열어서 확인")
    print(f"2. 라벨링 진행 (label 컬럼에 0/1/2 입력)")
    print(f"3. Colab 노트북에 업로드하여 학습")


if __name__ == '__main__':
    main()
