import random, Globals
from Firm import Firms 
from collections import deque
from Space import Space

class Workers:
    def __init__(
        self, space,  worker_id,
        position, num_positions
    ):
        self.space = space
        self.worker_id = worker_id
        self.employment_status = 0
        self.pay_off = 0
        self.position = position
        self.num_positions = num_positions


        # List to track vacancies and best offers
        self.vacancy_list = []
        self.best_offer_list = []

    def update_iteration(self, iteration):

        self.current_iteration = iteration


    # Worker applying behavior for bidding model
    def applying_bid(self):
        if Globals.model_type == 1:
            # Add this worker to the application list of all firms
            all_firms = self.space.get_objects(Firms)
            for firm in all_firms:
                firm.application_list.append(self)
    # Worker applying behavior for take-it-or-leave-it model
    def applying_takeit(self):
        if Globals.model_type == 0:
            # Add firms with vacancies to the vacancy list
            all_firms = self.space.get_objects(Firms)
            self.vacancy_list = list(all_firms)
            random.shuffle(self.vacancy_list)

            temp_payoff_current = 0

            for firm in self.vacancy_list:
                distance = min(
                    abs(self.position - firm.position),
                    self.position + self.num_positions - firm.position,
                    firm.position + self.num_positions - self.position
                )

                temp_payoff_new = (
                    firm.wage_offer - firm.wage_offer * Globals.fee * Globals.share_fee
                    - distance * Globals.effort
                )

                if temp_payoff_new >= temp_payoff_current:
                    self.best_offer_list.clear()
                    self.best_offer_list.append(firm)
                    temp_payoff_current = temp_payoff_new
                    self.pay_off = temp_payoff_current

            if self.best_offer_list:
                best_firm = self.best_offer_list[0]
                best_firm.application_list.append(self)

            self.best_offer_list.clear()
            self.vacancy_list.clear()

    # Accept offer
    def workers_accepts(self):

        self.vacancy_list = []
        self.best_offer_list = []

        if Globals.model_type == 1:
            all_firms = self.space.get_objects(Firms)

            for firm in all_firms:
                distance = min(
                    abs(self.position - firm.position),
                    self.position + self.num_positions - firm.position,
                    firm.position + self.num_positions - self.position
                )

                temp_payoff = (
                    firm.wage_offer - firm.wage_offer * Globals.fee * Globals.share_fee
                    - distance * Globals.effort
                )

                if firm.wage_offer >= temp_payoff:
                    self.vacancy_list.append(firm)

            random.shuffle(self.vacancy_list)

            temp_payoff_current = 0

            for firm in self.vacancy_list:
                distance = min(
                    abs(self.position - firm.position),
                    self.position + self.num_positions - firm.position,
                    firm.position + self.num_positions - self.position
                )

                temp_payoff_new = (
                    firm.wage_offer - firm.wage_offer * Globals.fee * Globals.share_fee
                    - distance * Globals.effort
                )

                if temp_payoff_new >= temp_payoff_current:
                    self.best_offer_list.clear()
                    self.best_offer_list.append(firm)
                    temp_payoff_current = temp_payoff_new
                    self.pay_off = temp_payoff_current

            if self.best_offer_list:
                best_firm = self.best_offer_list[0]
                best_firm.employment_list.append(self)
                self.where_work = best_firm.firm_id
                self.employment_status = 1

            self.best_offer_list.clear()
            self.vacancy_list.clear()