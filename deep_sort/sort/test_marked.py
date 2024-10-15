import numpy as np
import copy

def checkResolv_conflict(m1, m2):
    '''
    to check and resolve if there is conflict in the marked matrix
    m1: marked matrix \n
    m2: original zero entry matrix'''
    ind_zerCol = []
    for i in range(m1.shape[1]):
        col = m1[:,i]
        onesPos = np.where(col==1)[0]
        if len(onesPos)==0:
            ind_zerCol.append(i)
        elif len(onesPos) > 1:
            ind_matchingRow = (onesPos[-1])   
    try: ind_matchingRow
    except: ind_matchingRow = None
    if len(ind_zerCol) > 0 and ind_matchingRow is not None:
        orig_row = m2[ind_matchingRow]
        onesPos = np.where(orig_row==1)[0]
        for i in range(m2.shape[0]):
            row = m2[i]
            if row[onesPos].all():
                if m1[i,ind_zerCol[0]] == 0 and m2[i,ind_zerCol[0]] == 1:
                    change_row = m1[i,:]
                    _onesPos = np.where(change_row==1)[0]
                    if _onesPos in onesPos:
                        change_row[:] = 0
                        change_row[ind_zerCol[0]] = 1
                        m1[i,:] = change_row
                        m1[ind_matchingRow, :] = 0
                        m1[ind_matchingRow, _onesPos[0]] = 1
                        break
    return m1            

def markMatrix(matrix):
    _mat = copy.deepcopy(matrix)
    #iterate over each row to change the marked matrix
    a = 0
    for i in range(matrix.shape[0]):
        row = matrix[i,:]
        onesPos = np.where(row==1)[0] # check position of ones
        # keep only single one in a row if multiple entries are found
        if onesPos.shape[0]>1:
            if i==0:
                row[onesPos[1:]] = 0 # keep first instance and make rest zeros
                matrix[i,:] = row       # update
            elif i>0:
                # updatedPos stores positions of 'ones' where there is no 'one' in the 
                # same position in all the previous rows and a next row
                updatedPos = []     
                for j in range(onesPos.shape[0]): # positione of 'ones' in the current row
                    check = []  # to check whether current 'one' can be retained or not
                    # to check previous positions
                    for k in range(i):
                        if (matrix[i-(k+1),onesPos[j]]) == 0:
                            check.append(False)
                        else:
                            check.append(True)    
                    if i==matrix.shape[0]:
                        if not any(check):
                            updatedPos.append(onesPos[j])
                    elif (i+1 < matrix.shape[0]):
                        if not any(check) and (matrix[i+1,onesPos[j]]==0):
                            updatedPos.append(onesPos[j])
                    else:
                        if not any(check):
                            updatedPos.append(onesPos[j])       
                if len(updatedPos) == 0:
                    for j in range(onesPos.shape[0]):
                        a -= 1
                        if (matrix[i+a,a] == 0):
                            updatedPos.append(onesPos[a])
                            break    
                updatedRow = np.zeros_like(row)
                updatedRow[updatedPos[0]] = 1
                matrix[i,:] = updatedRow
    matrix = checkResolv_conflict(matrix, _mat)
    return matrix

if __name__ == '__main__':
    matrix = np.array([[1,0,0,0,0],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [1,0,1,1,0]])   
    matrix = markMatrix(matrix)
    print()