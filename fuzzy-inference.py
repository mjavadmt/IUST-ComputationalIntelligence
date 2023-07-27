# Q2_graded
# Do not change the above line.
from simpful import *

FS = FuzzySystem()

age_terms = ['very_young', 'young', 'middle', 'old', 'very_old']
age_splitter = AutoTriangle(5, terms=age_terms,
                            universe_of_discourse=[17, 36])

price_terms = ['extremely_cheap', 'very_cheap', 'cheap', 'middle', 'expensive', 'very_expensive',
               'extremely_expensive']
price_splitter = AutoTriangle(7,
                              terms=price_terms, universe_of_discourse=[20, 1000])

last_five_performance = AutoTriangle(3, terms=['bad', 'moderate', 'good'], universe_of_discourse=[0, 5])

position_list = ["GK", "RB", "CB1", "CB2", "LB", "DMF", "CMF", "AMF", "LWF", "CF", "RWF"]


def term_maker(team, identifier, is_price=False):
    if is_price:
        return f"{team}_{identifier}_{is_price}"
    return f"{team}_performance"


def add_variables_each_player(identifier):
    # add sepahan
    FS.add_linguistic_variable(f"sepahan_{identifier}_price", price_splitter)
    FS.add_linguistic_variable(f"sepahan_{identifier}_age", age_splitter)
    # add foolad
    FS.add_linguistic_variable(f"foolad_{identifier}_price", price_splitter)
    FS.add_linguistic_variable(f"foolad_{identifier}_age", age_splitter)


def add_variable_team():
    FS.add_linguistic_variable("foolad_performance", last_five_performance)
    FS.add_linguistic_variable("sepahan_performance", last_five_performance)


def add_total_variables():
    for pos in position_list:
        add_variables_each_player(pos)
    add_variable_team()


def add_outcome():
    result = AutoTriangle(3, terms=['sepahan_win', 'tie', 'foolad_win'], universe_of_discourse=[-10, 10])
    FS.add_linguistic_variable("result", result)


def make_rule(terms, values, outcome):
    make_str = "IF "
    for index, term in enumerate(terms):
        make_str += f"({term} IS {values[index]})"
        if index != len(terms) - 1:
            make_str += " AND "
    make_str += f" THEN (result IS {outcome})"
    return make_str


def add_variable_rules():
    rules = [
        make_rule([term_maker('foolad', 'LB', 'price')], ["extremely_expensive"], "foolad_win"),
        make_rule([term_maker("sepahan", "RB", "price"), term_maker("sepahan", "LB", "price")],
                  ["expensive", "expensive"],
                  "sepahan_win"),
        make_rule([term_maker("sepahan", "CMF", "age")], ["young"],
                  "sepahan_win"),
        make_rule([term_maker("sepahan", "")], ["good"], "tie"),
        make_rule([term_maker('foolad', ''), term_maker('sepahan', '')], ["good", "good"], "tie"),
        make_rule([term_maker('foolad', ''), term_maker('sepahan', '')], ["moderate", "moderate"], "tie"),
        make_rule([term_maker('foolad', ''), term_maker('sepahan', '')], ["bad", "bad"], "tie"),
        make_rule([term_maker('foolad', 'CB1', 'age')], ["very_old"], "sepahan_win"),
        make_rule([term_maker('foolad', 'CB2', 'age')], ["very_young"], "foolad_win"),
        make_rule([term_maker("sepahan", "RB", "price"), term_maker("sepahan", "LB", "price")],
                  ["expensive", "expensive"],
                  "sepahan_win"),
        make_rule(
            [term_maker("sepahan", "LWF", "age"), term_maker("sepahan", "RWF", "age"),
             term_maker("sepahan", "CF", "age")],
            ["young", "young", "middle"],
            "sepahan_win"),
        make_rule(
            [term_maker("sepahan", "GK", "price"), term_maker("foolad", "GK", "price")],
            ["expensive", "cheap"],
            "sepahan_win"),
    ]

    FS.add_rules(rules, verbose=True)


sepahan_values = [(270, 70), (338, 23), (428, 24), (405, 25), (383, 22), (225, 32), (225, 29), (585, 29), (496, 28),
                  (405, 22), (405, 31)]

foolad_values = [(405, 33), (108, 25), (698, 29), (450, 29), (315, 31), (585, 20), (495, 29), (450, 32), (563, 29),
                 (270, 33), (540, 27)]


def add_variable_values():
    FS.set_variable(term_maker("sepahan", ""), 4)
    FS.set_variable(term_maker("foolad", ""), 3)
    for index, value in enumerate(foolad_values):
        FS.set_variable(term_maker("foolad", position_list[index], "price"), value[0])
        FS.set_variable(term_maker("foolad", position_list[index], "age"), value[1])
        FS.set_variable(term_maker("sepahan", position_list[index], "price"), sepahan_values[index][0])
        FS.set_variable(term_maker("sepahan", position_list[index], "age"), sepahan_values[index][1])


if __name__ == "__main__":
    add_total_variables()
    add_outcome()
    add_variable_rules()
    add_variable_values()
    outcome_result = FS.inference()
    print("-----------------------------------------------------------")
    if outcome_result["result"] < 0:
        print("Sepahan 1 - 0 Foolad")
    else:
        print("Sepahan 0 - 1 Foolad")
    print("-----------------------------------------------------------")

# Remove this comment and type your codes here

