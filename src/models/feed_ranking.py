"""
Feed ranking algorithm for the election simulation.
"""

class FeedRankingAlgorithm:
    """Base feed ranking algorithm that ranks content based on friends' engagement."""
    
    def rank(self, agent, feed_contents, current_time):
        """
        Ranks content based on friends' engagement with amplified weight.
        
        Args:
            agent: The agent viewing the feed
            feed_contents: List of content items to rank
            current_time: Current simulation time
            
        Returns:
            Sorted list of content items
        """
        feed_contents.sort(
            key=lambda x: (100 if x.get('author_id') in agent.following else 0) +  # Prioritize content from followed users
                         (agent.friend_like_counts.get(x['content_id'], 0) +
                          agent.friend_reply_counts.get(x['content_id'], 0)) * 10 +
                         x['total_engagement'],
            reverse=True)
        return feed_contents 