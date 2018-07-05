#ifndef MENOH_NAIVE_COMPUTATION_NODE
#define MENOH_NAIVE_COMPUTATION_NODE

namespace menoh_impl {
    namespace naive_backend {

        class computation_node {
        public:
            virtual ~computation_node() = default;

            void run() { do_run(); }

        private:
            virtual void do_run() = 0;
        };

    } // namespace naive_backend
} // namespace menoh_impl

#endif // MENOH_NAIVE_COMPUTATION_NODE
