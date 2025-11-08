"""
데이터베이스 모델
PostgreSQL 스키마에 맞춘 SQLAlchemy 모델
"""

from datetime import datetime

from sqlalchemy import CheckConstraint, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Press(Base):
    """언론사 테이블"""

    __tablename__ = "press"

    press_id = Column(Integer, primary_key=True, autoincrement=True)
    press_name = Column(String(100), nullable=False, unique=True)

    # Relationships
    articles = relationship("Article", back_populates="press")

    def __repr__(self):
        return f"<Press(id={self.press_id}, name='{self.press_name}')>"


class Article(Base):
    """기사 테이블"""

    __tablename__ = "article"

    article_id = Column(Integer, primary_key=True, autoincrement=True)
    press_id = Column(Integer, ForeignKey("press.press_id", ondelete="CASCADE"), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    url = Column(Text, nullable=False, unique=True)
    published_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    press = relationship("Press", back_populates="articles")
    stance_analysis = relationship("StanceAnalysis", back_populates="article", uselist=False)
    topic_mappings = relationship("TopicArticleMapping", back_populates="article")

    def __repr__(self):
        return f"<Article(id={self.article_id}, title='{self.title[:30]}...')>"


class Topic(Base):
    """토픽 테이블"""

    __tablename__ = "topic"

    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_date = Column(DateTime, nullable=False)
    topic_rank = Column(Integer, nullable=False)
    main_article_id = Column(Integer, ForeignKey("article.article_id", ondelete="SET NULL"))
    article_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (CheckConstraint("topic_rank >= 1 AND topic_rank <= 7", name="check_topic_rank"),)

    # Relationships
    main_article = relationship("Article", foreign_keys=[main_article_id])
    article_mappings = relationship("TopicArticleMapping", back_populates="topic")
    recommended_articles = relationship("RecommendedArticle", back_populates="topic")

    def __repr__(self):
        return f"<Topic(id={self.topic_id}, rank={self.topic_rank}, date={self.topic_date})>"


class TopicArticleMapping(Base):
    """토픽-기사 매핑 테이블"""

    __tablename__ = "topic_article_mapping"

    mapping_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topic.topic_id", ondelete="CASCADE"), nullable=False)
    article_id = Column(Integer, ForeignKey("article.article_id", ondelete="CASCADE"), nullable=False)
    similarity_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("similarity_score >= 0 AND similarity_score <= 1", name="check_similarity_range"),
    )

    # Relationships
    topic = relationship("Topic", back_populates="article_mappings")
    article = relationship("Article", back_populates="topic_mappings")

    def __repr__(self):
        return f"<TopicArticleMapping(topic_id={self.topic_id}, article_id={self.article_id})>"


class StanceAnalysis(Base):
    """스탠스 분석 테이블"""

    __tablename__ = "stance_analysis"

    analysis_id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("article.article_id", ondelete="CASCADE"), nullable=False, unique=True)
    support_prob = Column(Float, nullable=False)  # 옹호 확률
    neutral_prob = Column(Float, nullable=False)  # 중립 확률
    oppose_prob = Column(Float, nullable=False)  # 비판 확률
    final_stance = Column(String(10), nullable=False)  # 최종 스탠스: 'support', 'neutral', 'oppose'
    confidence_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("support_prob >= 0 AND support_prob <= 1", name="check_support_prob"),
        CheckConstraint("neutral_prob >= 0 AND neutral_prob <= 1", name="check_neutral_prob"),
        CheckConstraint("oppose_prob >= 0 AND oppose_prob <= 1", name="check_oppose_prob"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_score"),
        CheckConstraint("final_stance IN ('support', 'neutral', 'oppose')", name="check_final_stance"),
    )

    # Relationships
    article = relationship("Article", back_populates="stance_analysis")

    def __repr__(self):
        return f"<StanceAnalysis(article_id={self.article_id}, stance='{self.final_stance}', confidence={self.confidence_score})>"


class RecommendedArticle(Base):
    """추천 기사 테이블"""

    __tablename__ = "recommended_article"

    recommendation_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topic.topic_id", ondelete="CASCADE"), nullable=False)
    article_id = Column(Integer, ForeignKey("article.article_id", ondelete="CASCADE"), nullable=False)
    stance_type = Column(String(10), nullable=False)  # 'support', 'neutral', 'oppose'
    recommendation_rank = Column(Integer, nullable=False)  # 1, 2, 3
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("stance_type IN ('support', 'neutral', 'oppose')", name="check_stance_type"),
        CheckConstraint("recommendation_rank >= 1 AND recommendation_rank <= 3", name="check_recommendation_rank"),
    )

    # Relationships
    topic = relationship("Topic", back_populates="recommended_articles")
    article = relationship("Article")

    def __repr__(self):
        return (
            f"<RecommendedArticle(topic_id={self.topic_id}, article_id={self.article_id}, stance='{self.stance_type}')>"
        )
