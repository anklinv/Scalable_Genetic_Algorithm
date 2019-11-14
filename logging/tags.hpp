// log tag definitnion version, change it when tags are modified
#define LOGGING_VERSION 0

#define LOGGING_TAG_VERSION 0

#define LOGGING_TAG_CLOCKS_PER_SEC 1

// -----------------------------------------------------------------------------
// wall clock events
// -----------------------------------------------------------------------------

#define LOGGING_TAG_WC_LOGGING_BEGIN            0x01000000
#define LOGGING_TAG_WC_LOGGING_END              0x01000001

#define LOGGING_TAG_WC_RANK_INDIVIDUALS_BEGIN   0x01000100
#define LOGGING_TAG_WC_RANK_INDIVIDUALS_END     0x01000101

#define LOGGING_TAG_WC_EPOCH_BEGIN              0x01000200
#define LOGGING_TAG_WC_EPOCH_END                0x01000201

#define LOGGING_TAG_WC_BREED_POPULATION_BEGIN   0x01000300
#define LOGGING_TAG_WC_BREED_POPULATION_END     0x01000301

#define LOGGING_TAG_WC_MUTATE_POPULATION_BEGIN  0x01000400
#define LOGGING_TAG_WC_MUTATE_POPULATION_END    0x01000401


// -----------------------------------------------------------------------------
// CPU clock events
// -----------------------------------------------------------------------------

#define LOGGING_TAG_CC_LOGGING_BEGIN   0x02000000
#define LOGGING_TAG_CC_LOGGING_END     0x02000001

// -----------------------------------------------------------------------------
// values
// -----------------------------------------------------------------------------

#define LOGGING_TAG_VAL_BEST_FITNESS    0x03000000
