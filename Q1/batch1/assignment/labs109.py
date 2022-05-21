#Accending List
from array import *
def is_ascending (item):
    flag = True
    i = 1
    while i < len(item): 
        if(item[i] <= item[i - 1]): 
            flag = False
        i += 1
    return flag

#Riffle
import math
from array import *
def riffle(items, out=True):
    part2=math.floor(len(item)/2)
    i=0
    while(part2<len(item)):
        if(out==True):
            print(item[i],item[part2])
        else:
            print(item[part2],item[i])
        i+=1;
        part2+=1;

#Only_odd_digits
def only_odd_digits(n):
    integers=str(n)
    flag=True
    for i in integers:
        if(int(i)%2==0):
            flag=False
    return flag

#Cyclops Numbers
import math
def is_cyclops(n):
    num=str(n)
    flag=True
    mid=math.floor(len(num)/2)
    if(num[mid]!='0'):
        flag=False
        return flag
    else:
        if(len(num)==1):
            flag=True
            return flag
        else:
            i=0
            mid+=1
            while(mid<len(num)):
                if((num[i]=='0') or (num[mid]=='0')):
                    flag=False
                    return flag                   
                i+=1
                mid+=1
                return flag

#Domino Cycle
def domino_cycle(tiles):
    length=len(tiles)
    flag=0
    for i in range(0,1):
        if(tiles[i][1]!=tiles[(i+1)%length][0]):
            flag=1
            break
        return (True if flag==0 else False)

def count_growlers(animals):
  #this variable stores the growling count
  count = 0
  #traversing through the list
  for j,i in enumerate(animals):
    temp = 0
    #if is is cat or dog that means we have to check left of the list
    if(i == 'cat' or i == 'dog'):
      #checking from left of the i
      for k in range(j):
      #if it is a dog or god temp is incremented
        if(animals[j-k-1] == 'dog' or animals[j-k-1] == 'god'):
              temp += 1
        #if any element is a cat or a tac tem is decreased
        else:
            temp -= 1         
    #same here if i is tac or god we will check to the right of it
    elif(i == 'tac' or i == 'god'):
  #checking for dog or god and incrementing temp
      for k in range(len(animals) - j-1):
        if(animals[j+k+1] == 'dog' or animals[j+k+1] == 'god'):
              temp += 1
        #decrementing if cat or tac is found
        else:
              temp -= 1
  #if after checking more number of dog or god were found the temp would be greater than zero
    
    count += 1

      
    #count is increased
            
  #returning the count
  return count

def scylla_or_charybdis(sequence, n):
    #steps follows the distnace travelled from the begining and will help determine when youve fallen off
    steps = 0
    #count follows the amount of moves it takes to fall off
    count = 0
    #print(f"You have n - 1 = {n-1} step(s) in both directions before you fall off")
    k = 1
    j = 1
    items = {}
    while k != len(sequence) // 2 + 1:
        #since steps is guaranteed to end with 2n consecutve steps to the right
        for i in range(0, len(sequence), k):
            if j*k-1 > len(sequence)-1:
                k += 1
                steps = 0
                count = 0
                j = 1
                break
            if sequence[j*k-1] == "+":
                steps += 1
                count += 1
                j += 1
            elif sequence[j*k-1] == "-":
                steps -= 1
                count += 1
                j += 1
            if abs(steps) == n:
                items[k] = count
                #print(f"You have fallen due to reaching step {steps}")
                #print(f"using every {k}th step of the sequence it took only {count} moves to fall off")
                k = k + 1
                steps = 0
                count = 0
                j = 1
                break

#Count Dominators
def count_dominators(items):
    i=0
    count=1
    if(len(items)==0):
        return 0
    elif(len(items)==1):
        return count
    else:
        while(i<len(items)-1):
            Yes=True
            j=i+1
            while(j<len(items)):
                if(items[i]<=items[j]):
                    Yes=False
                    break
                j+=1
            if(Yes==True):
                count+=1
            i+=1  
        return count

#Extracting Increasing integers form digit string
def extract_increasing(digits):
    count=1
    res=[]
    if(len(digits)==0):
        return
    elif(len(digits)==1):
        return res.append(int(digits(0)))
    else:
        for idx in range(0, len(digits), count):
            if(int(digits)):            
                digits='1'
    extract_increasing(digits)





def words_with_letters(words, letters):
    result=[]

    for letter in letters:
        words_list=[]
        letter=list(letter)   #converting the letter to list of characters

        for word in words:   #checking all words
            letter_index=0   #index of letter
            equal_count=0   #no of characters of the letter found in the word
            word=list(word)   #converting the word to list of characters

            for i in range(0,len(word)):
                if(word[i]==letter[letter_index]):
                    if(letter_index==len(letter)-1):   #if all characters of letter found in word
                        words_list.append("".join(word))   #convert list to string and append to the words_list
                        break
                    equal_count+=1
                    letter_index+=1
        result.append(words_list)   #append the words list to the final result list
    return result

#Running Median of Three
def running_median_of_three(items):
    result = []
    for i in range(len(items)):
        if i < 2:
            result.append(items[i])
        else:
            if (items[i - 2] <= items[i - 1] <= items[i]) or (items[i] <= items[i - 1] <= items[i - 2]):
                median = items[i - 1]
            elif (items[i - 2] <= items[i] <= items[i - 1]) or (items[i - 1] <= items[i] <= items[i - 2]):
                median = items[i]
            else:
                median = items[i - 2]
            result.append(median)
    return result

#The card that wins the trick
def winning_card(cards,trump = None):
    rank_dict={'deuce':2, 'trey':3, 'four':4, 'five':5, 'six':6,'seven':7,
    'eight':8, 'nine':9, 'ten':10, 'jack':11, 'queen':12, 'king':13,
    'ace':14}
    if trump is not None:
        selected_cards = [card for card in cards if card[1]==trump]    
    else:
        selected_cards = cards
        max_card = max(selected_cards,key = lambda card: rank_dict[card[0]])
    return max_card

#Boustrophedon
def create_zigzag(rows, cols, start = 1):
    result = []
    for i in range(rows):
        tmp = []
        for j in range(cols):
            tmp.append(start)
            start += 1
        if(i%2==1):
            tmp.reverse()
        result.append(tmp)
    return result

#Multidimentional Knight Moves
def move(knight, start, end):
    new_knight = []
    length = len(knight)
    for i in range(length):
        new_knight.append(knight[i])
    for i in range(length):
        value = abs(start[i] - end[i])
        if value in new_knight:
            new_knight.remove(value)
            continue
        else:
            return False
    return True

def winning_card(card,trump=None):
  
  #list to determine rank of card
  l=["one","two","three","four","five","six","seven","eight","nine","ten","jack","queen","king","ace"]
  
  #if trump is none, take trump as first card type
  if trump == None:
    s = card[0][1]
    #first card will be trump
    p=0
  else:
    s = trump
    #find position of first trump card
    for i in range(len(card)):
      if card[i][1] == s:
        p=i
        break
  
  #iterate through the card list
  #which card has higher power in the type of trump
  for i in range(len(card)):
    if card[i][1] == s:
      #greater index defines higher rank
      if l.index(card[i][0]) >= l.index(card[p][0]):
        p = i
  
  #print the card
  print(card[p])

#Seven ruls, zeros Drool
def seven_zero(n):
    d = 1 
    ans = 0
    while True:
        if n%2 == 0 or n%5 == 0:
            k = 1
            while k <= d:
                val = int(k * '7' + (d-k) * '0') 
                if val%n == 0:
                    ans = val
                    break
                k += 1
        else:
            val = int(d * '7')
            ans = val if val%n == 0 else 0
        d += 1
        if ans > 0:
             return ans
             break

#Sum of the balls off the brass monkey
def phyramid_block(n,m,h):
  sum1=0;
  for i in range(h):
    sum1=sum1+(n*m)
    n=n+1
    m=m+1
  return sum1

#FulCrum
def can_balance(items):
    if len(items)==1:
        return 0
    for i in range(1,len(items)-2):
        lans = 0
        for j in range(i):
            lans = lans + (i-j)*items[j]
        rans = 0
        for j in range(i+1,len(items)):
            rans = rans + (j-i)*items[j]
        if lans == rans:
            return i
    return -1

#Last Man standing
def find_seq(n,k):
        mens=[(i+1) for i in range(n)]
        st=0
        seq=[]
        sz=n
        while len(mens)>1:
                st=(st+(k-1))%sz
                sz-=1
                seq.append(mens[st])
                del mens[st]
        seq.append(mens[0])
        return seq

def three_summers(items, goal):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            for k in range(j + 1, len(items)):
                if items[i] + items[j] + items[k] == goal:
                    return True
    return False

def crag_score(dice):
    dice.sort()
    if dice[0]==dice[1] and dice[1]==dice[2]:return 25
    elif sum(dice)==13:
        if dice[0]==dice[1] or dice[0]==dice[2] or dice[1]==dice[2]:
            return 50
        else:
            return 26
    elif dice[0]==4 and dice[1]==5 and dice[2]==6:return 20
    elif dice[0]==1 and dice[1]==2 and dice[2]==3:return 20
    elif dice[0]==2 and dice[1]==4 and dice[2]==6:return 20
    elif dice[0]==1 and dice[1]==3 and dice[2]==5:return 20
    else : return max(dice)

#Longest palindrome substring
def longest_palindrome(text):
    longest = ''
    for i in range(len(text)):
        for j in range(i+1, len(text)+1):
            if text[i:j][::-1] == text[i:j]:
                if len(text[i:j]) > len(longest):
                    longest = text[i:j]
    return longest

#All your fractions are belong to base
def group_and_skip(n,out,inp):
    remainingCoin=[]
    while(n>out):
        n=n-out
        remainingCoin.append(n%out)
        n=int(n/out)+inp
    remainingCoin.append(n)
    print(remainingCoin)

import statistics


def tukeys_ninthers(items):

    #Declare sublists as type of list

    sublists = []

    #calculate length of items

    lengthofitems = len(items)

    #Iterate loop

    for each in range(0,lengthofitems,3):

        #append to sublists list

        sublists.append(items[each:each+3])

    #Declare medianNewList as type of list

    medianNewList = []

    #Iterate loop

    for eachsub_list in sublists:

        #calculate medianValue value

        medianValue = int(statistics.median(eachsub_list))

        #append the element medianValue to medianNewList

        medianNewList.append(medianValue)

       

    #calculate lengthofmedian

    lengthofmedian = len(medianNewList)

    #check lengthofmedian is less than equal to 3

    if lengthofmedian <= 3:
      
        turnElement = statistics.median(medianNewList)
        return turnElement

    else:

        #call recursive approach

        turnElement = tukeys_ninthers(medianNewList)

        return turnElement

#Recaman's Sequence
def recman(n):
    sequence=[1]
    hash_set=set([1])
    for i in range(1,n,1):
        val=sequence[i-1]-(i+1)
        if val<=0 or val in hash_set:
            val=sequence[i-1]+i+1
        sequence.append(val)
        hash_set.add(val) 
    return sequence

def fibonacci_word(k):
    
    if k==0:
        return '0'
    if k==1:
        return '1'
    
    
    a="0"
    b="1"
    result=""
    
    while len(result)!=k+1:
            
        result=result+a+b
        a=b
        b=result
      
        if len(result)>=k:
            break
    
    
    return result[k]

#Rversing the Reversed
def reverse_reversed(items):

    reversed = [] 

    for item in items[::-1]:

        if isinstance(item, list): 

            reversed.append(reverse_reversed(item)) 

        else:

            reversed.append(item) 

    return reversed

#count word dominators
# definition of the function.

def dominates(w1, w2):

    cnt = 0

    # start the for loop

    for (c1, c2) in zip(w1, w2):

        # check the condition.

        if c1 > c2:

             # update the value.

            cnt += 1

    # return the value.

    return cnt > len(w1)/2

# definition of the function.

def count_word_dominator(words):

    cnt = 0

    for i in range(len(words)):

        dominateStatus = True

        for j in range(i + 1, len(words)):

            if not dominates(words[i], words[j]):

                dominateStatus = False

        if dominateStatus:

            cnt += 1

    return cnt

#Duplicates digit bonus
def rpt(n):
    x=list(n)
    x.reverse()
    f=0
    cost=0
    k=-1

    last=x[0]

    #looks for repeating digits
    for i in range(len(x)):

        k=-1
        if x[i]=='#':
            continue
        c=x[i]
        x[i]='#'
        #Calculate the value for the repeating digits
        for j in range(i+1,len(x)):

            if x[j]==c:
                k=k+1
                x[j]='#'
            else:
                break
        if k>=0:
            if f==0 and c==last:
                cost+=(2*(10)*k)
                f=1
            else:
                cost=cost+((10)*k)


    return cost

#Nearest Smaller element

def nearest_smaller(items):

    result = []

    curr_ele_index = 0

    while len(result) < len(items):

        smallest_ele_left = 'x'

        smallest_ele_right = 'x'

        left_index = curr_ele_index - 1

        while left_index >= 0:

            if(items[left_index] < items[curr_ele_index]):

                smallest_ele_left = left_index

                break

            left_index -= 1

        right_index = curr_ele_index + 1

        while right_index < len(items):

            if(items[right_index] < items[curr_ele_index]):

                smallest_ele_right = right_index

                break

            right_index += 1

        if smallest_ele_left == 'x':

            if smallest_ele_right == 'x':

                result.append(items[curr_ele_index])

            else:

                result.append(items[smallest_ele_right])

        else:

            if smallest_ele_right == 'x':

                result.append(items[smallest_ele_left])

            else:

                dist_left = curr_ele_index - smallest_ele_left

                dist_right = smallest_ele_right - curr_ele_index

                if(dist_left < dist_right):

                    result.append(items[smallest_ele_left])

                elif (dist_right < dist_left):

                    result.append(items[smallest_ele_right])

                else:

                    smaller_ele = min(items[smallest_ele_left], items[smallest_ele_right])

                    result.append(smaller_ele)

        curr_ele_index += 1

    return result

def collatzy_distance(start, end):
    currentSeq = [start]

    dis = 0
    while end not in currentSeq:
        nextSeq = []
        for c in currentSeq:
            nextSeq.append(3*c + 1)
            nextSeq.append(c//2)
        currentSeq = nextSeq
        dis += 1

    return dis

#Interesting Intersecting
def squares_intersect(s1,s2):
    x1,x2,y1,y2,r1,r2=s1[0],s2[0],s1[1],s2[1],s1[2],s2[2]
    if ((x1<x2) and (x1+r1<x2)) or ((y1<y2) and (y1+r1<y2)):
        return False
    elif ((x2<x1) and (x2+r2<x1)) or ((y2<y1) and (y2+r2<y1)):
        return False
    else:
        return True

def give_change(amount, coins):
    change = [] # store the change
    j = 0; # to index the coins
    while amount > 0:
        if amount >= coins[j]:
            amount = amount - coins[j]
            change.append(coins[j])
        else:
            j = j + 1
    return change



#Keep doubling
def double_untill_all_digits(n, giveup = 1000):
    counter = 0
    numbers = [0,1,2,3,4,5,6,7,8,9]
    digits = [int(i) for i in str(n)]
    for i in range(giveup):
        if all(elem in digits for elem in numbers):
            return counter
        else:
            n = n * 2
            counter += 1
            digits = [int(i) for i in str(n)]
            return -1

#that's enough of you
def remove_kth(items,k):

    unique_items = list(set(items))
    count_items = [0]*len(unique_items)
    final_items = []
    for i in range(len(items)):
        ind = unique_items.index(items[i])
        count_items[ind] = count_items[ind]+1
        if count_items[ind]<=k: 
            final_items = final_items + [items[i]] 
        return final_items

#Longest palindrome substring
def longest_palindrome(text):
    longest = ''
    for i in range(len(text)):
        for j in range(i+1, len(text)+1):
            if text[i:j][::-1] == text[i:j]:
                if len(text[i:j]) > len(longest):
                    longest = text[i:j]
    return longest

#All your fractions are belong to base
def group_and_skip(n,out,inp):
    remainingCoin=[]
    while(n>out):
        n=n-out
        remainingCoin.append(n%out)
        n=int(n/out)+inp
    remainingCoin.append(n)
    print(remainingCoin)

#Count Consective summers
def count_consecutive_summers(n):
    count = 0
    for i in range(1, n+1):
        total = 0
        for j in range(i, n+1):
            total += j
            if total == n:
                count += 1
    return count



#Pulldonw Your Neighbours
def eliminate_neighbours( items ): 
    items_list = items.copy( )
    size = len( items_list )
    larger = 0; smaller = items[0]
    neighbour = 0
    small_ind = 0;
    count = 0
    for i in items_list:
        if i > larger: 
            larger = i 
        if ( larger <= size ):
            while ( larger in items_list ):
                size = len( items_list )
                for j in items_list:
                    if j < smaller: 
                        smaller = j 
                        small_ind = items_list.index( j ) 
                    if ( small_ind == 0 ):
                        neighbour = items_list[ small_ind +1 ]
                    elif ( small_ind == size-1 ):
                        neighbour = items_list [ small_ind -1 ] 
                    else: 
                        if ( items_list[ small_ind -1 ] > items_list [ small_ind +1 ] ):
                            neighbour = items_list[ small_ind -1 ] 
                        else:
                            neighbour = items_list[ small_ind +1 ] 
            items_list.remove( smaller )
            items_list.remove( neighbour ) 
            count += 1
            smaller = larger
        return ( count )

#What do you hear, what do you say
def count_and_say(digits):

    if len(digits) ==0:

        return ""

    digitletters =[]

    digitCount = []

    previousDigit =''

   

    count =-1

    for digit in digits:

        if digit == previousDigit:

            digitCount[count] += 1

           

        else:
            digitletters.append(digit)

            count +=1

            digitCount.append(1)

            previousDigit = digit

   

    output =""

    count =0

    for digit in digitletters:

        output+= str(digitCount[count]) + digit

        count +=1
        
    return output

def safe_squares_rooks(n,rooks):
    #creating two sets for storing unsafe rows and cols
    unsafe_rows=set()
    unsafe_cols=set()
    #looping through rooks list
    for rook in rooks:
        #adding current rook's row and col to unsafe row,col sets
        #sets prevent duplicates, so we dont need to worry about them
        unsafe_rows.add(rook[0])
        unsafe_cols.add(rook[1])
    #number of safe rows or cols = n - number of unsafe rows or cols
    safe_rows_count=n-len(unsafe_rows)
    safe_cols_count=n-len(unsafe_cols)
    #returning total number of safe spaces
    return safe_rows_count*safe_cols_count

#Bishops on a binge
def safe_squares_bishops(n, bishops):
    safe_cells = 0
    for row in range(n):
        for col in range(n):
            safe=True
            for pos in bishops:
                if abs(row - pos[0]) == abs(col - pos[1]):
                    safe=False
                    break
            if safe:safe_cells+=1
    return safe_cells



#Up for the count
def counting_series(n):
    len = 1;
    count = 9;
    start = 1;
    while n > len * count :
        n= n - len * count
        len = len + 1
        count = count * 10
        start = start * 10
    start = start + (n - 1) / len
    s = str(start)
    return s[((n - 1) % len)]

#Reverse the vowels
def reverse_vowels(text):
    vowels=[]
    for c in text:
        i=c.lower()
        if i=='a' or i=='e' or i=='i' or i=='o' or i=='u':
            vowels.append(c)
    vowels.reverse()
    k=0
    str=""
    for j in range(len(text)):
        i=text[j].lower()
        if i=='a' or i=='e' or i=='i' or i=='o' or i=='u':
            st=str+vowels[k]
            k+=1
        else:
            str=str.text[j]
    str=str[0].upper()+str[1:]
    return str

#Everybody do a scrooge shuffle
def spread_the_coins(pile,left,right):
    start=i=0
    while i <(len(pile)):
        k = pile[i]//(left+right)
        if k>0:
            pile[i]-=k*(left+right)
            if i!=len(pile)-1:
                pile[i+1] += k*right
            else:
                pile.append(k*right)
            if i!=0:
                pile[i-1] += k*left
                i-=1
                continue
            else:
                start-=1
                pile.insert(0,k*left)
                continue
        i+=1
    return(start,pile)

#Calking-wilf sequence
def calkin_wilf(n):

    queue=[]
    f=Fraction(1,1)
    queue.append(f)
    ans=Fraction(1,1)
    while n>0:
        temp=queue.pop(0)

        ans=temp

        p=temp.numerator
        q=temp.denominator

        f1=Fraction(p,p+q)
        f2=Fraction(p+q,q)

        queue.append(f1)
        queue.append(f2)
        n=n-1
        return ans

#Hippity hoppity, abolish loopity
def frog_collision(frog1, frog2):
    s1_x, s1_y, d1_x, d1_y = frog1[0], frog1[1], frog1[2], frog1[3]
    s2_x, s2_y, d2_x, d2_y = frog2[0], frog2[1], frog2[2], frog2[3]
    try:
        t1 = (s1_x - s2_x)/(d2_x - d1_x)
        t2 = (s1_y - s2_y)/(d2_y - d1_y)
        if t1 == t2:
            return t1
        else:
            return None
    except:
        if d2_x - d1_x == 0 and s1_x - s2_x == 0 and d2_y - d1_y != 0:
            return (s1_y - s2_y)/(d2_y - d1_y)
        elif d2_y - d1_y == 0 and s1_y - s2_y == 0 and d2_x - d1_x != 0:
            return (s1_x - s2_x)/(d2_x - d1_x)
        else:
            return None

#Double Trouble
def double_trouble(items,n):
    data = []
    counter,value = -1,1
    while True:
        counter = (counter+1)%len(items)
        data.append(value)
        if sum(data) >= n: break
        if counter == len(items) -1 : value *= 2
    return items[counter]

#Nearest Polygonal Number
def fun(s, i):
    return ((s-2)*(i*i) - (s-4)*i)//2

def nearest_poligonal_number(n, s):
    a = 1
    b = 2
    while fun(s, b) < n:
        b = b*b
    m = (a+b)//2
    while b-a >= 2:
        at_m = fun(s,m)
        if at_m > n:
            b = m
        elif at_m < n:
             a = m
        else:
            return at_m
        m = (a+b)//2
    at_a = fun(s,a)
    at_b = fun(s,b)
    if abs(at_a-n) <= abs(at_b-n):
            return at_a
    else:
            return at_b

#Postfix Interpreter
from collections import deque
def postfix_evaluate(items):
    stack = deque()

    for x in items:
        if x=='+' or x=='-' or x=='*' or x=='/':
            op1=stack.pop()
            op2=stack.pop()
            result=calculate(op2,op1,x)
            stack.append(result)
        else:
            stack.append(x)
         
    return stack.pop()

def calculate(op2,op1,x):
    if x is '*':
        return int(op2)*int(op1)
    elif x is '/':
        if (op1 == 0):
            return 0
        else:
            return int(op2)//int(op1)
    elif x is '+':
        return int(op2)+int(op1)
    elif x is '-':
        return int(op2)-int(op1)

#Fractran Interpreter
import sys
def fractran(n:int, prog:list, giveup:int=1000):
    # Intialize result list with n
    res = [n]
    # Run the logic, giveup number of times
    while giveup:
        # A flag which is set if a integer calue is found
        found = False
        for x in prog:
            frac = n * float(x[0])/float(x[1])
            # If n * fraction is an integer, add it to result list,
            # change n to this integer and break the loop to proceed to next iteration
            if frac.is_integer():
                n = int(frac)
                res.append(n)
                found = True
                break
        # If flag is not set terminate the program
        if not found:
            sys.exit(-1)
        giveup -= 1
    return res

#Permutation Cycyle
def is_permutation(items, n):
    found = []
    for i in range(1, n + 1):
        for j in range(0, len(items)):
            if items[j] == i:
                found.append(i)
                break
    if len(found) == n:
        return True
    else:
        return False

#Ztalloc ecneqes
def ztalloc(shape):
    prev=1

    for i in range(len(shape)-1,-1,-1): 
        if(shape[i]=='d'): 
            prev=prev*2
        else: 
            prev=prev-1
            if((prev)%3!=0):
                return None
            else: 
                prev=prev/3   
                if(((prev)%2==0)):
                    return None
    return int(prev)

#Reverse ascending sublist
def reverse_ascending_sublists(items):
    low = 0
    final = []
    for i in range(1, len(items)-1):
        if items[i-1] >= items[i]:
            temp = items[low:i]
            final += temp[::-1]
            low = i
    temp = items[low:]
    if items[-1] > items[-2]:
        final += temp[::-1]
    else:
        final += temp
    return final

#Brangelina
def brangelina(first, second): 
    vowels = ['a','e','i', 'o', 'u']
    fcnt = 0
    sindex = 0
    findex = 0
    flag = False
    for i in range(len(first)): # This for-loop counts the number of vowel groups
        if first[i] in vowels:
            if not flag:
                fcnt += 1
                findex = i      # set vowel group index
                flag = True
        else:
            flag = False
            
    if fcnt > 1:                # if vowel group count >0
        flag = False
        for i in range(findex-1, -1, -1):
            if first[i] in vowels:
                findex = i      # find the index of second last vowel group
                flag = True
            else:
                if flag:
                    break
    for i in range(len(second)):  # find the index of first vowel group
        if second[i] in vowels:
            break
        sindex +=1
    return(first[:findex] + second[sindex:]) # return combined word

#Uambcselne the word
def unscramble(words,word):
    guesses=[]
    for i in words:
        if i[0]==word[0] and i[-1]==word[-1] and len(word)==len(i):
            if sorted(list(word[1:-1]))==sorted(list(i[1:-1])):
                guesses.append(i)
    return guesses

def pancake_scramble(text):
#Initialize variable
    s = text
      #Loop from 2 to n
    for i in range(2,len(text)+1):
      #Reverse first i chars and keep the next chars as is
        s = s[:i][::-1]+s[i:]
    return s

#Substituition words
def substitution_words(seed):
    rng = random.Random(seed)
    f = open('words_alpha.txt', 'r', encoding='utf-8')
    words = [x.strip() for x in f]
    f.close()
    yield ('ABCD', words)
    for i in range(100):
        pat = ''
        ll = int(sqrt(rng.randint(3*3, 10*10)))
        n = rng.randint(1, ll)
        for j in range(ll):
            pat += ups[rng.randint(0, n - 1)]
            yield (pat, words)

#Mahattan Skyline
def manhattan_skyline(towers):   
    if not towers:
        return []
    if len(towers) == 1:
        return [[towers[0][0], towers[0][2]], [towers[0][1], 0]]

    mid = len(towers) // 2
    left = manhattan_skyline(towers[:mid])
    right = manhattan_skyline(towers[mid:])
    return merge(left, right)

def merge(left, right):
    h1, h2 = 0, 0
    i, j = 0, 0
    result = []

    while i < len(left) and j < len(right):
        if left[i][0] < right[j][0]:
            h1 = left[i][1]
            corner = left[i][0]
            i += 1
        elif right[j][0] < left[i][0]:
            h2 = right[j][1]
            corner = right[j][0]
            j += 1
        else:
            h1 = left[i][1]
            h2 = right[j][1]
            corner = right[j][0]
            i += 1
            j += 1
        if is_valid(result, max(h1, h2)):
            result.append([corner, max(h1, h2)])
    result.extend(right[j:])
    result.extend(left[i:])
    return result

#Count Overlapping Disks
def count_overlapping_disks_generator(seed):
    rng = random.Random(seed)
    for n in range(3, 150):
        d = 2 * n
    for i in range(10):
        disks = set()
        while len(disks) < n:
            x = rng.randint(-d, d)
            y = rng.randint(-d, d)
            r = rng.randint(1, n)
            disks.add((x, y, r))
        disks = list(disks)
        disks.sort()
        yield (disks,)

#an Unordinary ordering of ordinary cardinals
def sort_by_digit_count(seed):
    rng = random.Random(seed)
    for k in range(1000):
        n = k + 2
        yield ([rng.randint(1, n * n * n * n) for i in range(n)],)

#Count Visible in range
def count_divisibles_in_range(start, end, n):

    count = 0;


    for i in range(start, end+1):
        if (i % n == 0):
            count = count + 1

    return count

#Bridge hand shape
def bridge_hand_shape(hand):
    count = [0]*4
    for x in hand:
        if x[1] == 'spades':
            count[0]+=1
        elif x[1] == 'hearts':
            count[1]+=1
        elif x[1] == 'diamonds':
            count[2]+=1
        else:
            count[3]+=1
    return count

#Milton work point count
def cards(hand, trump = 'notrump'):
    suits = {'spades':0,'hearts':0,'diamonds':0,'clubs':0}
    points = 0
    for card in hand:
        value, suit = card
        if value == 'ace':
            points += 4
        elif value == 'king':
            points += 3
        elif value == 'queen':
            points += 2
        elif value == 'jack':
            points += 1
        suits[suit] += 1
    if sorted(suits.values()) == [3, 3, 3, 4]:
        points -= 1
        return points
    for suit, n in suits.items():
        if n >= 7:
            points += 3
        elif n == 6:
            points += 2
        elif n == 5:
            points += 1
        if trump != 'notrump' and suit != trump:
            if n == 0:
                points += 5
            elif n == 1:
                points += 3
    return points

#Bulls and cows
def bullCow(orgNum,predNum,bulls,cows):
    bi = 0
    ci =0
    orgNum = str(orgNum)
    predNum = str(predNum)
    for i in range(4):
        if orgNum[i] == predNum[i]:
            bi += 1
        elif orgNum[i] in predNum:
            ci+=1
    return (bi==bulls) and(ci==cows)

#Sort by more frequency
from collections import Counter
def frequency_sort(elems):
    result = [item for items, c in Counter(elems).most_common()
    for item in [items] * c]
    return result

#Calling all units,BandE in progress
def is_perfect_power(n):
    if n <= 1:
        return True

    # loop from 2 to square root
    for i in range(2, int(n ** 0.5) + 1):
        f = i
        while f <= n:
            f = f * i

            if f == n:
                return True

    return False

#Fabonacci sum
def fibonacci_sum(n):
    first = 0
    second = 1
    fib_list = []
    values = []
    fib_list.append(first)
    fib_list.append(second)
    next = first + second
    # Constructing a list of fibonacci numbers upto n
    while next <= n:
        fib_list.append(next)
        first = second
        second = next
        next = first + second
    answer = 0
    fib_list.sort(reverse=True)
    # Taking a decision based on value of answer
    while True:
        for i in fib_list:
            if answer + i > n:
                continue
            else:
                values.append(i)
                answer = answer + i
                if answer == n:
                    break
                fib_list.remove(i)
        if answer == n:
            break

    return values

#Rocks with Friends
def rooks_with_friends(n,friends,enemies):
    chess=[]
    for i in range(n):
        temp=[]
        for j in range(n):
            temp.append(0)
        chess.append(temp)
    for x,y in friends:
        chess[x][y]=1
    for x,y in enemies:
        chess[x][y]=-1
    ans=0
    for i in range(n):
        for j in range(n):
            if chess[i][j]==1 or chess[i][j]==-1:
                continue
            f=0
            for k in range(j+1,n):
                if chess[i][k]==-1:
                    f=1
                    break
                if chess[i][k]==1:
                    break
                for k in range(j-1,-1,-1):
                    if chess[i][k]==-1:
                        f=1
                        break
                if chess[i][k]==1:
                    break
        for k in range(i-1,-1,-1):
            if chess[k][j]==-1:
                f=1
                break
            if chess[k][j]==1:
                break
        for k in range(i+1,n):
            if chess[k][j]==-1:
                f=1
                break
            if chess[k][j]==1:
                break
        if f==0:
            ans+=1
    return ans

#possible words in hangman
def possible_words_generator(seed):
  f = open('words_sorted.txt', 'r', encoding='utf-8')
  words = [x.strip() for x in f]
  f.close()
  rng = random.Random(seed)
  letters = 'abcdefghijklmnopqrstuvwxyz'
  for i in range(100):
    k = rng.randint(1, 10)
    guessed = set(rng.sample(letters, k))
    patword = rng.choice(words)
    pat = ''
    for ch in patword:
      pat += ch if ch in guessed else '*'
    yield (words, pat)

def get_word_shape(word ) :
    ## initialize the word shape to empty list
    wordShape = []
    ## iterate thru the word length
    for i in range( 1 , len(word ) ) :
        ## Compare the current with previous letter
        ## if the diff is > 0 append 1 to the word shape
        ## if the diff is < 0 append -1 to the word shape
        ## if the diff is 0 append 0 to the word shape
        ## ord(letter) returns the ascii code of that letter
        if ( ord(word[i]) - ord(word[i-1]) ) > 0 :
            wordShape.append(1)
        elif ( ord(word[i]) - ord(word[i-1]) ) < 0 :
            wordShape.append(-1)
        else :
            wordShape.append(0)
    ## finally return the wordshape
    return wordShape

#Factoring Factorials
def factoring_factorial(n):
  if n < 3:
    return [(n, 1)]
  count = dict()
  for i in range(1, n+1):
    f = prime_factors(i)
    for val in f:
      count[val] = count.get(val, 0) + 1
  ans = [(p,e) for p,e in count.items()]
  return ans

#Aliqout Sequence
def getProperDivisor(N):
  sum = 0
  for i in range(1, int(sqrt(N)) + 1):
    if N % i == 0:
      if N // i == i:
        sum += i
      else:
        sum += i
        sum += N // i
  return sum - N

def aliquot_sequence(n,giveup=100):
  lst = []
  lst.append(n);
  giveup=giveup-1
  while n > 0 and giveup>=1:
    n = getProperDivisor(n)
    if n in lst:
      break
    lst.append(n);
    giveup=giveup-1
  return lst;

def possible_words(words,pattern):
  matches=[]
  for word in words:
    isMatch=True
    if len(word)==len(pattern):
      for i in range(len(pattern)):
        if pattern[i].isalpha():
          if pattern[i]!=word[i]:
            isMatch=False
            break
          else:
            if word[i] in pattern:
              isMatch=False
              break
      if isMatch:
        matches.append(word)
  return matches

#All paths leads to Rome
def lattice_paths(m, n, tabu):
  if m == 0 and n == 0:
    return 1
  if m < 0 or n < 0:
    return 0
  x = [tup[0] for tup in tabu]
  y = [tup[1] for tup in tabu]
  if m in x and n in y:
    return 0
  return lattice_paths(m - 1, n, tabu) + lattice_paths(m, n - 1, tabu)

def collatzy_distance(start, end):
    currentSeq = [start]

    dis = 0
    while end not in currentSeq:
        nextSeq = []
        for c in currentSeq:
            nextSeq.append(3*c + 1)
            nextSeq.append(c//2)
        currentSeq = nextSeq
        dis += 1

    return dis

def isSorted(points):
  for i in range(0,len(points)-1):
    if points[i][0]+points[i][1]>points[i+1][0]+points[i+1][1]:
      return False
    return True
def count_maximal_layers(points):
  n=len(points)
  if n==0:
    return 0
  #Distance metric (x+y) based increasing sort
  if not isSorted(points):
    for i in range(0,n):
      for j in range(i+1,n):
        if points[i][0]+points[i][1]>points[j][0]+points[j][1]:
          points[i],points[j]=points[j],points[i]
  #New list containing non-maximal layer points
  non_maximal_layer_points=[]
  for i in range(0,n):
    for j in range(i+1,n):
      if points[i][0]<points[j][0] and points[i][1]<points[j][1]:
        non_maximal_layer_points.append(points[i])
        break
  #Add 1 in the final result of how many times maximal layer points should be removed
  return 1 + count_maximal_layers(non_maximal_layer_points)

def square_follows(it):
    result=[]
    prev_values=set()
    for num in it:
        sq_rt=num**0.5
        if sq_rt in prev_values:
            result.append(int(sq_rt))
        prev_values.add(num)
    return result

list = []
def sum_of_distinct_cubes(n):
    if n==0 :
        return True

    for x in range(n , 0, -1):
        if(sum_of_distinct_cubes(n - (x*x*x)) == True):
            if(x not in list):
                list.append(x)
                return True
    return False


if(sum_of_distinct_cubes(8) == True):
    print(list[::-1])
else:
    print(None)

def perimeter_limit_split(a,b,p):
  dp=[[None]*(b+1) for i in range(a+1)]#dp array for memoization
  return perimeter_limit_split_dp(a,b,p,dp)#calling helper function
  #helper function
def perimeter_limit_split_dp(a,b,p,dp):
  #base case
  if p>=2*(a+b):
    return 0
  m1=m2=M1=M2=0
  m=float('inf')
  #recurring cases
  if a>1:
    for i in range(1,a//2+1):
      m1 = perimeter_limit_split_dp(i,b,p,dp)if dp[i][b]==None else dp[i][b]
      m2 = perimeter_limit_split_dp(a-i,b,p,dp) if dp[a-i][b]==None else dp[a-i][b]
      m = min(1+m1+m2,m)
  if b>1:
    for i in range(1,b//2+1):
      m1 = perimeter_limit_split_dp(a,i,p,dp) if dp[a][i]==None else dp[a][i]
      m2 = perimeter_limit_split_dp(a,b-i,p,dp) if dp[a][b-i]==None else dp[a][b-i]
      m = min(1+m1+m2,m)
  #memoizing in dp table
  dp[a][b]=m
  return m

def van_eck(n):
  prev_index = [-1]*(n+1)
  curr_index = [-1]*(n+1)
  count = [0]*(n+1)
  arr = [0]
    
  curr_index[arr[-1]] = 0
  count[arr[-1]]+=1
    
  for i in range(1,n+1):
    if count[arr[-1]]==1:
      arr.append(0)
      count[arr[-1]]+=1
      prev_index[arr[-1]] = curr_index[arr[-1]]
      curr_index[arr[-1]] = len(arr)-1
    else:
      arr.append(curr_index[arr[-1]] - prev_index[arr[-1]])
      count[arr[-1]] +=1
      prev_index[arr[-1]] = curr_index[arr[-1]]
      curr_index[arr[-1]] = len(arr)-1
  return arr[-1]



def balanced_ternary(n):
   l=[]
   i=0
   while n!=0:
       if n%3==2 or n%3==-1:
           l.append(-1*pow(3,i))
       elif n%3==1 or n%3==-2:
           l.append(pow(3,i))
       if n<0:
           n=int((n-1)/3)
       else:
           n=int((n+1)/3)
       i+=1
   return l[::-1] #reversing order of exponents

def bulgarianSolitaire(piles, k, moves):

  reqdConfiguration = [i+1 for i in range(k)]

  piles.sort()

  if(piles == reqdConfiguration):

    return moves

  piles = [x - 1 for x in piles]

  piles.append(len(piles))

  piles = list(filter(lambda a: a != 0, piles))

  moves += 1

  # print(moves,piles)

  return bulgarianSolitaire(piles,k,moves)

def recurse(letters, n, tabu, cur_pos, current_string, final_list):
      # base case
    if cur_pos == n:
        final_list.append(current_string)
        return

      # we can put all the letters from letters in this position if it does not voilate the substring condition
    for letter in letters:
        new_str = current_string+letter
        allowed = True
        for item in tabu:
            if len(new_str) >= len(item):
                index = -1*len(item)
                if new_str[index:] == item:
                    allowed = False
                    break
        if allowed:
            recurse(letters, n, tabu, cur_pos+1, new_str, final_list)


def forbidden_substring(letters, n, tabu):
    letters = [char for char in letters]
    final_list = []
    letters.sort()
    recurse(letters, n, tabu, 0, "", final_list)
    return final_list

import numpy as np
import platform
def md ( a = 3, b = 10, n = 2, dt = 0.1 ):
    step_print_index = 0
    step_print_num = 10
    step_print = 0

    for step in range ( 0, step_num + 1 ):

        if ( step == 0 ):
            pos, vel, acc = initialize ( p_num, d_num )
        else:
            pos, vel, acc = update ( p_num, d_num, pos, vel, force, acc, mass, dt )

        force, potential, kinetic = compute ( p_num, d_num, pos, vel, mass )

        if ( step == 0 ):
            e0 = potential + kinetic

        if ( step == step_print ):
            rel = ( potential + kinetic - e0 ) / e0
            print ( '  %8d  %14f  %14f  %14g' % ( step, potential, kinetic, rel ) )
            step_print_index = step_print_index + 1
            step_print = ( step_print_index * step_num ) // step_print_num

    return

def unscramble(words,word):
    guesses=[]
    for i in words:
        if i[0]==word[0] and i[-1]==word[-1] and len(word)==len(i):
            if sorted(list(word[1:-1]))==sorted(list(i[1:-1])):
                guesses.append(i)
    return guesses

def eliminate_neighbours( items ): 
    items_list = items.copy( )
    size = len( items_list )
    larger = 0; smaller = items[0]
    neighbour = 0
    small_ind = 0;
    count = 0
    for i in items_list:
        if i > larger: 
            larger = i 
        if ( larger <= size ):
            while ( larger in items_list ):
                size = len( items_list )
                for j in items_list:
                    if j < smaller: 
                        smaller = j 
                        small_ind = items_list.index( j ) 
                if ( small_ind == 0 ):
                    neighbour = items_list[ small_ind +1 ]
                elif ( small_ind == size-1 ):
                    neighbour = items_list [ small_ind -1 ] 
                else: 
                    if ( items_list[ small_ind -1 ] > items_list [ small_ind +1 ] ):
                        neighbour = items_list[ small_ind -1 ] 
                    else:
                        neighbour = items_list[ small_ind +1 ] 
                items_list.remove( smaller )
                items_list.remove( neighbour ) 
                count += 1
                smaller = larger
            return ( count )

def seven_zero(n):
    d = 1 
    ans = 0
    while True:
        if n%2 == 0 or n%5 == 0:
            k = 1
            while k <= d:
                val = int(k * '7' + (d-k) * '0') 
                if val%n == 0:
                    ans = val
                    break
                k += 1
        else:
            val = int(d * '7')
            ans = val if val%n == 0 else 0
        d += 1
        if ans > 0:
             return ans
             break

def running_median_of_three(items):
    result = []
    for i in range(len(items)):
        if i < 2:
            result.append(items[i])
        else:
            if (items[i - 2] <= items[i - 1] <= items[i]) or (items[i] <= items[i - 1] <= items[i - 2]):
                median = items[i - 1]
            elif (items[i - 2] <= items[i] <= items[i - 1]) or (items[i - 1] <= items[i] <= items[i - 2]):
                median = items[i]
            else:
                median = items[i - 2]
            result.append(median)
    return result

def only_odd_digits(n):
    integers=str(n)
    flag=True
    for i in integers:
        if(int(i)%2==0):
            flag=False
    return flag

def double_untill_all_digits(n, giveup = 1000):
    counter = 0
    numbers = [0,1,2,3,4,5,6,7,8,9]
    digits = [int(i) for i in str(n)]
    for i in range(giveup):
        if all(elem in digits for elem in numbers):
            return counter
        else:
            n = n * 2
            counter += 1
            digits = [int(i) for i in str(n)]
            return -1