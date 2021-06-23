
class PandasComputations:

    """Useful pandas computations not in pandas"""


    def __init__(self, data_frame):
        self.dataframe = data_frame



    def check_df(self):
        print(self.dataframe.head())

    def cumulative_subtraction_across_column(self, column_name):
        list_of_cases = []
        series_items = list(self.dataframe[column_name])
        
        for i in range(len(series_items)):
            count = 0 
            try:
                    
                count = count + series_items[i+1] - series_items[i]
                    
            except IndexError:
                print('completed')

            
            final_count = count
            list_of_cases.append(final_count)


        

        









