'''
Instructions:
Calculate probability of getting a positive test result, P(Pos)
'''
# P(D)
p_diabetes = 0.01

# P(~D)
p_no_diabetes = 0.99

# Sensitivity or P(Pos|D)
p_pos_diabetes = 0.9

# Specificity or P(Neg/~D)
p_neg_no_diabetes = 0.9

p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1 - p_neg_no_diabetes)) 

print('The probability of getting a positive test result P(Pos) is: {}', format(p_pos))

'''
Instructions:
Compute the probability of an individual having diabetes, given that, that individual got a positive test result.
In other words, compute P(D|Pos).

The formula is: P(D|Pos) = (P(D) * P(Pos|D) / P(Pos)
'''

# P(D|Pos)
p_diabetes_pos = ( p_diabetes * p_pos_diabetes ) / p_pos

print('the probability having diabetes, given that, that individual got a positive test result is :\
', format(p_diabetes_pos))

'''
Instructions:
Compute the probability of an individual not having diabetes, given that, that individual got a positive test result.
In other words, compute P(~D|Pos).

The formula is: P(~D|Pos) = (P(~D) * P(Pos|~D) / P(Pos)

Note that P(Pos/~D) can be computed as 1 - P(Neg/~D). 

Therefore:
P(Pos/~D) = p_pos_no_diabetes = 1 - 0.9 = 0.1
'''

# P(Pos/~D)
p_pos_no_diabetes = 0.1 # p_pos_no_diabetes = 1 - p_neg_no_diabetes

# P(~D|Pos)
p_no_diabetes_pos = (p_no_diabetes * p_pos_no_diabetes) / p_pos

print('Probability of an individual not having diabetes, given that that individual got a positive test result is:\
', format(p_no_diabetes_pos))