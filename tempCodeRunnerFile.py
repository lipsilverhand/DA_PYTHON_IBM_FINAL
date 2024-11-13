sns.regplot(x='bmi', y='charges', data=df, line_kws={"color": "red"})
plt.ylim(0,)
plt.show()