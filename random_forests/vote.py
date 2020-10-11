# votes = list of votes
def get_majority_vote (votes):
	votes_table = {} # New hash table
	for vote in votes:
	    if vote in votes_table:    # Check if key in hash table
	        votes_table[vote] += 1 # Increment counter
	    else:
	        votes_table[vote] = 1  # Create counter for vote
	# Find max counter in hash table
	return max(votes_table, key=votes_table.get)