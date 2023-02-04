def kelly_criterion(bets):
    bankroll = 100 # starting bankroll
    for bet in bets:
        home_odds = bet['home_odds']
        draw_odds = bet['draw_odds']
        away_odds = bet['away_odds']
        # calculate probability for each outcome
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds
        # calculate the sum of probabilities
        prob_sum = home_prob + draw_prob + away_prob
        # normalize the probabilities
        home_prob = home_prob / prob_sum
        draw_prob = draw_prob / prob_sum
        away_prob = away_prob / prob_sum
        # calculate the best outcome
        best_outcome = max(home_prob*home_odds, draw_prob*draw_odds, away_prob*away_odds)
        # calculate the fraction
        fraction = (best_outcome - 1) / best_outcome
        # calculate the bet size
        bet_size = fraction * bankroll
        print(f'Bet size for bet with home_odds {home_odds}, draw_odds {draw_odds} and away_odds {away_odds}: {bet_size}')

bets = [{'home_odds': 2.0, 'draw_odds': 3.0, 'away_odds': 4.0}, {'home_odds': 1.5, 'draw_odds': 3.5, 'away_odds': 5.0}]
kelly_criterion(bets)
